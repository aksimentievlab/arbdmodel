# -*- coding: utf-8 -*-
## Test with `python -m arbdmodel.dry_martini`
"""
Dry Martini v2.1 (implicit-solvent Martini) lipid model.

Arnarez et al., JCTC 2015, 11, 260-275, doi:10.1021/ct500477k

Topology and parameters are read from the bundled GROMACS .itp files
(resources/dry_martini). Nonbonded: LJ force-switched 9-12 A; Coulomb
cut at 12 A with epsilon_r = 15 (potential-shift by default).
"""

import numpy as np

## Local imports
from .logger import logger, get_resource_path
from . import ArbdModel, ParticleType, PointParticle, Group
from .interactions import AbstractPotential, HarmonicBond, CosineAngle

KJ_TO_KCAL = 1/4.184
NM_TO_AA = 10.0
COULOMB_CONST = 332.06371       # kcal/mol AA / e^2

DEFAULT_ITP_FILES = ("dry_martini/dry_martini_v2.1.itp",
                     "dry_martini/dry_martini_v2.1_lipids.itp")

__all__ = ['DryMartiniTopology', 'DryMartiniNonbonded', 'DryMartiniLipid',
           'DryMartiniModel', 'read_gro_lipids']


## ---------------------------------------------------------------------------
## Topology
## ---------------------------------------------------------------------------
class _Molecule:
    def __init__(self, name, nrexcl):
        self.name = name
        self.nrexcl = nrexcl
        self.atoms = []         # (name, type, charge)
        self.bonds = []         # (i, j, b0 [AA], k [kcal/mol/AA^2])
        self.angles = []        # (i, j, k, theta0 [deg], k [kcal/mol])
        self.unsupported = {}   # section -> line count


class DryMartiniTopology:
    """
    Minimal GROMACS .itp reader for Dry Martini.

    Supports #define, #ifdef/#ifndef/#else/#endif, [ defaults ], [ atomtypes ],
    [ nonbond_params ], [ moleculetype ], [ atoms ], [ bonds ] (funct 1) and
    [ angles ] (funct 2). Other molecule sections are recorded and rejected
    when that molecule is built.

    Parameters
    ----------
    itp_files : sequence of str, optional
        Paths to .itp files, read in order. Defaults to the bundled Dry Martini v2.1 files.
    defines : sequence of str, optional
        Preprocessor symbols treated as defined.
    """

    def __init__(self, itp_files=None, defines=()):
        if itp_files is None:
            itp_files = [get_resource_path(f) for f in DEFAULT_ITP_FILES]
        self.defines = {d: '' for d in defines}
        self.masses = {}            # type -> amu
        self.lj = {}                # (typeA, typeB) -> (sigma [AA], epsilon [kcal/mol])
        self.molecules = {}         # name -> _Molecule
        self._comb_rule = None
        for f in itp_files:
            self._read(f)

    @staticmethod
    def _strip(line):
        line = line.split(';')[0]
        if not line.lstrip().startswith('#'):
            line = line.split('#')[0]
        return line.strip()

    def _expand(self, tokens):
        out = []
        for t in tokens:
            out.extend(self.defines[t].split() if t in self.defines else [t])
        return out

    def _read(self, filename):
        section, mol, skip = None, None, []
        with open(filename) as fh:
            for raw in fh:
                line = self._strip(raw)
                if not line:
                    continue

                if line.startswith('#'):
                    words = line.split()
                    key = words[0]
                    if key in ('#ifdef', '#ifndef'):
                        defined = words[1] in self.defines
                        skip.append(not defined if key == '#ifdef' else defined)
                    elif key == '#else':
                        skip[-1] = not skip[-1]
                    elif key == '#endif':
                        skip.pop()
                    elif any(skip):
                        pass
                    elif key == '#define':
                        vals = []
                        for w in words[2:]:
                            if w.startswith('#'): break
                            vals.append(w)
                        self.defines[words[1]] = ' '.join(vals)
                    elif key == '#include':
                        logger.debug(f'DryMartiniTopology: ignoring {line} in {filename}')
                    continue
                if any(skip):
                    continue

                if line.startswith('['):
                    section = line.strip('[] ').lower()
                    continue

                tok = self._expand(line.split())
                if section == 'defaults':
                    self._comb_rule = int(tok[1])
                elif section == 'atomtypes':
                    self.masses[tok[0]] = float(tok[1])
                elif section == 'nonbond_params':
                    if self._comb_rule != 2:
                        raise NotImplementedError('Only comb-rule 2 (sigma, epsilon) is supported')
                    a, b = tok[0], tok[1]
                    sigma = float(tok[3]) * NM_TO_AA
                    eps = float(tok[4]) * KJ_TO_KCAL
                    self.lj[(a, b)] = self.lj[(b, a)] = (sigma, eps)
                elif section == 'moleculetype':
                    mol = _Molecule(tok[0], int(tok[1]))
                    self.molecules[mol.name] = mol
                elif section == 'atoms':
                    mol.atoms.append((tok[4], tok[1], float(tok[6])))
                elif section == 'bonds':
                    i, j, funct = int(tok[0])-1, int(tok[1])-1, int(tok[2])
                    if funct != 1:
                        raise NotImplementedError(f'{mol.name}: bond funct {funct}')
                    b0 = float(tok[3]) * NM_TO_AA
                    kb = float(tok[4]) * KJ_TO_KCAL / NM_TO_AA**2
                    mol.bonds.append((i, j, b0, kb))
                elif section == 'angles':
                    i, j, k, funct = int(tok[0])-1, int(tok[1])-1, int(tok[2])-1, int(tok[3])
                    if funct != 2:
                        raise NotImplementedError(f'{mol.name}: angle funct {funct}')
                    mol.angles.append((i, j, k, float(tok[4]), float(tok[5]) * KJ_TO_KCAL))
                elif mol is not None and section is not None:
                    mol.unsupported[section] = mol.unsupported.get(section, 0) + 1

    def molecule(self, name):
        """ Return the parsed molecule `name`, rejecting unsupported terms. """
        try:
            mol = self.molecules[name]
        except KeyError:
            raise KeyError(f'{name} not found; available: {sorted(self.molecules)}')
        if mol.unsupported:
            raise NotImplementedError(f'{name}: unsupported sections {mol.unsupported}')
        if mol.nrexcl != 1:
            raise NotImplementedError(f'{name}: nrexcl={mol.nrexcl} (only 1 is supported)')
        return mol


## ---------------------------------------------------------------------------
## Nonbonded
## ---------------------------------------------------------------------------
def _shifted_inverse_power(r, a, r1, rc):
    """ GROMACS force-switched 1/r^a, switched between r1 and rc, zero beyond rc. """
    A = -a*((a+4)*rc - (a+1)*r1) / (rc**(a+2) * (rc-r1)**2)
    B =  a*((a+3)*rc - (a+1)*r1) / (rc**(a+2) * (rc-r1)**3)
    C = 1/rc**a - A/3*(rc-r1)**3 - B/4*(rc-r1)**4
    u = r**-a - C
    s = r > r1
    u[s] -= A/3*(r[s]-r1)**3 + B/4*(r[s]-r1)**4
    u[r > rc] = 0
    return u


class DryMartiniNonbonded(AbstractPotential):
    """
    Dry Martini nonbonded potential: force-switched LJ + cut-off Coulomb.

    Parameters
    ----------
    lj_params : dict
        (typeA_name, typeB_name) -> (sigma [AA], epsilon [kcal/mol]).
    r_switch : float
        LJ switching distance (AA).
    r_cut : float
        LJ and Coulomb cutoff (AA).
    epsilon_r : float
        Relative dielectric constant.
    coulomb : {'potential-shift', 'force-switch'}
        'potential-shift': q1 q2 / epsilon_r (1/r - 1/r_cut), as GROMACS
        coulomb-modifier = Potential-shift. 'force-switch': force switched
        over (0, r_cut], as GROMACS coulombtype = shift (template_sd.mdp).
    resolution : float
        Table spacing (AA).
    """

    def __init__(self, lj_params, r_switch=9.0, r_cut=12.0, epsilon_r=15.0,
                 coulomb='potential-shift', resolution=0.01):
        AbstractPotential.__init__(self, range_=(0, r_cut), resolution=resolution, zero='last')
        if coulomb not in ('potential-shift', 'force-switch'):
            raise ValueError(f'Unknown coulomb treatment {coulomb!r}')
        self.lj_params = lj_params
        self.r_switch = r_switch
        self.r_cut = r_cut
        self.epsilon_r = epsilon_r
        self.coulomb = coulomb

    def potential(self, r, types=None):
        typeA, typeB = types
        r = np.asarray(r, dtype=float)
        sigma, eps = self.lj_params[(typeA.name, typeB.name)]
        u = 4*eps * (sigma**12 * _shifted_inverse_power(r, 12, self.r_switch, self.r_cut)
                     - sigma**6 * _shifted_inverse_power(r, 6, self.r_switch, self.r_cut))
        qq = typeA.charge * typeB.charge
        if qq != 0:
            if self.coulomb == 'force-switch':
                phi = _shifted_inverse_power(r, 1, 0.0, self.r_cut)
            else:
                phi = np.where(r <= self.r_cut, 1/r - 1/self.r_cut, 0.0)
            u = u + COULOMB_CONST*qq/self.epsilon_r * phi
        u[~np.isfinite(u)] = np.nan       # r = 0; filled by AbstractPotential
        return u


## ---------------------------------------------------------------------------
## Lipids and model
## ---------------------------------------------------------------------------
class DryMartiniLipid(Group):
    """
    One Dry Martini molecule built from its .itp entry.

    Parameters
    ----------
    molecule : _Molecule
        Parsed molecule from DryMartiniTopology.molecule().
    coordinates : array_like, shape (n_atoms, 3)
        Bead positions (AA), in .itp atom order.
    types : dict
        Bead type name -> ParticleType.
    resid : int
        Residue id written to PSF/PDB.
    """

    def __init__(self, molecule, coordinates, types, resid=1, **kwargs):
        coords = np.asarray(coordinates, dtype=float)
        if coords.shape != (len(molecule.atoms), 3):
            raise ValueError(f'{molecule.name}: expected {len(molecule.atoms)}x3 coordinates, got {coords.shape}')
        Group.__init__(self, name=molecule.name, **kwargs)
        self.resname = molecule.name
        self.resid = resid

        beads = [PointParticle(types[t], r, name=name, resname=molecule.name, resid=resid)
                 for (name, t, q), r in zip(molecule.atoms, coords)]
        for b in beads:
            self.add(b)

        for i, j, b0, k in molecule.bonds:
            self.add_bond(beads[i], beads[j], _bond(k, b0), exclude=True)
        for i, j, l, theta0, k in molecule.angles:
            self.add_angle(beads[i], beads[j], beads[l], _angle(k, theta0))


_potential_cache = {}
def _bond(k, r0):
    key = ('bond', round(k, 6), round(r0, 6))
    if key not in _potential_cache:
        _potential_cache[key] = HarmonicBond(k=k, r0=r0, range_=(0, 50), resolution=0.01)
    return _potential_cache[key]

def _angle(k, theta0):
    key = ('cosangle', round(k, 6), round(theta0, 6))
    if key not in _potential_cache:
        _potential_cache[key] = CosineAngle(k=k, r0=theta0, resolution=0.01)
    return _potential_cache[key]


class DryMartiniModel(ArbdModel):
    """
    ARBD model of Dry Martini molecules.

    Parameters
    ----------
    lipids : sequence of (str, array_like)
        (molecule name, coordinates [AA] of shape (n_atoms, 3)) per molecule.
    dimensions : sequence of float
        Periodic box lengths (AA).
    damping_coefficient : float
        Langevin damping (1/ns). Default 250 = 1/tau_t with tau_t = 4 ps (template_sd.mdp).
    topology : DryMartiniTopology, optional
        Parsed force field; defaults to the bundled Dry Martini v2.1 files.
    coulomb : str
        Coulomb treatment passed to DryMartiniNonbonded.
    **conf_params
        Passed to ArbdModel (timestep, temperature, ...). cutoff defaults to the
        12 AA Dry Martini cutoff.
    """

    def __init__(self, lipids, dimensions, damping_coefficient=250, topology=None,
                 coulomb='potential-shift', **conf_params):
        logger.info("""You are using Dry Martini v2.1 as described in:
        C. Arnarez et al., Dry Martini, a coarse-grained force field for lipid membrane simulations with implicit solvent.
        J. Chem. Theory Comput. 2015, 11, 260-275. doi:10.1021/ct500477k
        
Please cite all appropriate articles!""")

        self.topology = DryMartiniTopology() if topology is None else topology
        if 'cutoff' not in conf_params:
            conf_params['cutoff'] = 12.0

        molecules = [(self.topology.molecule(name), xyz) for name, xyz in lipids]
        self.types = self._make_types([m for m, _ in molecules], damping_coefficient)
        children = [DryMartiniLipid(m, xyz, self.types, resid=i+1)
                    for i, (m, xyz) in enumerate(molecules)]

        ArbdModel.__init__(self, children, dimensions=dimensions, **conf_params)

        nonbonded = DryMartiniNonbonded(self.topology.lj, r_cut=conf_params['cutoff'], coulomb=coulomb)
        names = sorted(self.types)
        for i, a in enumerate(names):
            for b in names[i:]:
                self.add_nonbonded_interaction(nonbonded, typeA=self.types[a], typeB=self.types[b])

    def _make_types(self, molecules, damping_coefficient):
        charges = {}
        for mol in molecules:
            for name, t, q in mol.atoms:
                if charges.setdefault(t, q) != q:
                    raise ValueError(f'Bead type {t} has charges {charges[t]} and {q}; '
                                     'per-type charge is required')
        return {t: ParticleType(t, charge=q, mass=self.topology.masses[t],
                                damping_coefficient=damping_coefficient, resname=t)
                for t, q in charges.items()}

    @classmethod
    def from_gro(cls, gro_file, dimensions=None, **kwargs):
        """ Build a model from a GROMACS .gro file (box taken from the file unless given). """
        lipids, box = read_gro_lipids(gro_file)
        return cls(lipids, dimensions=box if dimensions is None else dimensions, **kwargs)


def read_gro_lipids(gro_file):
    """
    Read a .gro file into per-residue coordinates.

    Returns
    -------
    lipids : list of (str, ndarray)
        (residue name, coordinates [AA]) per residue, in file order.
    box : list of float
        Box lengths (AA).
    """
    with open(gro_file) as fh:
        lines = fh.read().splitlines()
    n = int(lines[1])
    lipids, key = [], None
    for line in lines[2:2+n]:
        k = (int(line[0:5]), line[5:10].strip())
        xyz = [float(line[20+8*d:28+8*d]) * NM_TO_AA for d in range(3)]
        if k != key:
            lipids.append((k[1], []))
            key = k
        lipids[-1][1].append(xyz)
    box = [float(x) * NM_TO_AA for x in lines[2+n].split()[:3]]
    return [(name, np.array(xyz)) for name, xyz in lipids], box


if __name__ == "__main__":
    top = DryMartiniTopology()
    print("MOLECULES", sorted(top.molecules))
    popc = top.molecule('POPC')
    for name, t, q in popc.atoms:
        print(f"{name}\t{t}\t{q:+.1f}\t{top.masses[t]}")
