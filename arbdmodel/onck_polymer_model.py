# -*- coding: utf-8 -*-
## Test with `python -m arbdmodel.onck_polymer_model`

import numpy as np
import sys

## Local imports
from .logger import devlogger, logger, get_resource_path
from .core_objects import ParticleType, PointParticle, Citation
from .polymer import PolymerBeads, PolymerModel
from .interactions import AbstractPotential, HarmonicBond, TabulatedPotential

"""Define particle types"""
_types = dict(
    A = ParticleType("ALA",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.7, # dimensionless
                     lambda_ = 0.72973,
                 ),
    R = ParticleType("ARG",
                     mass = 120,
                     charge = 1,
                     epsilon = 0.0,
                     lambda_ = 0.0,
                 ),
    N = ParticleType("ASN",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.33,
                     lambda_ = 0.432432,
                 ),
    D = ParticleType("ASP",
                     mass = 120,
                     charge = -1,
                     epsilon = 0.0005,
                 ),
    C = ParticleType("CYS",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.68,
                 ),
    Q = ParticleType("GLN",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.64,
                 ),
    E = ParticleType("GLU",
                     mass = 120,
                     charge = -1,
                     epsilon = 0.0005,
                 ),
    G = ParticleType("GLY",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.41,
                 ),
    H = ParticleType("HIS",
                     mass = 120,
                     charge = 0.0,
                     epsilon = 0.53,
                 ),
    I = ParticleType("ILE",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.98,
                 ),
    L = ParticleType("LEU",
                     mass = 120,
                     charge = 0,
                     epsilon = 1.0,
                 ),
    K = ParticleType("LYS",
                     mass = 120,
                     charge = 1,
                     epsilon = 0.0005,
                 ),
    M = ParticleType("MET",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.78,
                 ),
    F = ParticleType("PHE",
                     mass = 120,
                     charge = 0,
                     epsilon = 1.0,
                 ),
    P = ParticleType("PRO",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.65,
                 ),
    S = ParticleType("SER",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.45,
                 ),
    T = ParticleType("THR",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.51,
                 ),
    W = ParticleType("TRP",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.96,
                 ),
    Y = ParticleType("TYR",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.82,
                 ),
    V = ParticleType("VAL",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.94,
                 )
)
for k,t in _types.items():
    t.resname = t.name
    t.version = 1.0

_types_versions = {1.0: _types}

## https://link.springer.com/article/10.1007/s12274-022-4647-1
_types_versions['1.0cp'] = {k:ParticleType(t.name,
                                           mass = t.mass,
                                           charge = t.charge,
                                           epsilon = t.epsilon) for k,t in _types.items()}
for k in 'R D E K'.split(): _types_versions['1.0cp'][k].epsilon = 0.005

_types_versions[1.1] = dict()   # https://www.pnas.org/doi/10.1073/pnas.2221804120
__si_data = """A 71.07 0.7 0
L 113.16 1 0
R 156.19 0.005 +1
K 128.17 0.005 +1
N 114.10 0.41 0
M 131.20 0.78 0
D 115.09 0.005 -1
F 147.18 1 0
C 103.14 0.68 0
P 97.12 0.65 0
Q 128.13 0.33 0
S 87.08 0.45 0
E 129.11 0.005 -1
T 101.11 0.51 0
G 57.05 0.48 0
W 186.22 0.96 0
H 137.14 0.53 0
Y 163.18 0.82 0
I 113.16 0.98 0
V 99.13 0.94 0"""
for k, m, e, q in [l.split() for l in __si_data.split('\n')]:
    t0 = _types_versions[1.0][k]
    t  = _types_versions[1.1][k] = ParticleType(t0.name+'v1.1', mass=float(m),
                                                epsilon = float(e), charge = float(q), version=1.1)
    assert(t.charge == t0.charge)
    if t0.epsilon != t.epsilon:
        devlogger.info(f'Updating epsilon for 1BPA-1.1 {k}: {t0.epsilon} -> {t.epsilon}')

devlogger.info(f'Onck model version 1BPA-1.1 has {np.sum([t0.epsilon != _types_versions[1.1][k].epsilon for k,t0 in _types_versions[1.0].items()])} different epsilon parameters compared to 1BPA')

""" Cation-Pi interactions """
eps_cp_dict = {('ARG','PHE'): 4.3,
               ('ARG','TYR'): 5.0,
               ('ARG','TRP'): 6.7,
               ('LYS','PHE'): 1.79,
               ('LYS','TYR'): 3.13,
               ('LYS','TRP'): 4.26} # Table S2; kJ/mol
eps_cp_dict = {k:v*0.23900574 for k,v in eps_cp_dict.items()} # convert to kcal/mol
for k,v in list(eps_cp_dict.items()): eps_cp_dict[(k[1],k[0])] = v # add reversed keys to dictionary

version_name = {1.0:'1BPA', '1.0cp':'1BPA-CP', 1.1:'1BPA-1.1'}

__base_ref = Citation(
    author  = 'A. Fragasso, H.W. de Vries, J. Andersson, et al',
    title   = 'A designer FG-Nup that reconstitutes the selective transport barrier of the nuclear pore complex',
    journal = 'Nat Commun',
    volume  = 12,
    year    = 2021,
    doi     = '10.1038/s41467-021-22293-y'
)
__ref1 = Citation(
    author  = 'A. Fragasso, H.W. de Vries, J. Andersson, et al',
    title = 'Transport receptor occupancy in nuclear pore complex mimics',
    journal = 'Nano Res',
    volume = 15,
    pages = '9689–9703',
    year = 2022,
    doi = '10.1007/s12274-022-4647-1'
)
__ref2 = Citation(
    author="M. Dekker, E. Van der Giessen, and P.R. Onck",
    title="Phase separation of intrinsically disordered FG-Nups is driven by highly dynamic FG motifs",
    journal='PNAS',
    volume=120,
    number = 25,
    pages = 'e2221804120',
    year = 2023,
    doi = '10.1073/pnas.2221804120'
)

version_refs = {1.0: (__base_ref,),
                '1.0cp': (__base_ref, __ref1),
                1.1: (__base_ref, __ref2),
                }

## Default to the latest
_types = _types_versions[1.1]

class OnckNonbonded(AbstractPotential):
    """
    Nonbonded interaction potential for 1BPA and 1BPA-1.1.

    This class implements nonbonded interactions including electrostatics with
    distance-dependent dielectric and Lennard-Jones type potentials.

    Parameters
    ----------
    debye_length : float, optional
        The Debye-Hückel screening length in angstroms. Default is 12.7 Å.
    resolution : float, optional
        The spatial resolution for potential calculation. Default is 0.1 Å.
    range_ : tuple, optional
        The range of distances for which to calculate the potential. Default is (0, None).
    max_force : float
        Maximum allowed force in the system, used as a cap for stability. Default is 50

    Note
    -----
    The potential includes:
    
    1. Electrostatic interactions with distance-dependent dielectric constant
    2. Modified Lennard-Jones interactions that depend on residue types:
       - Special 8-6 LJ potential for cationic-aromatic interactions in 1.0cp/1.1 versions
       - General LJ-type potential for other interactions

    The effective dielectric D decreases with distance according to a specific formula.
    """

    """ Nonbonded interaction for 1BPA and 1BPA-1.1 """
    def __init__(self, debye_length=10/1.27, resolution=0.01, range_=(0,None)):
        AbstractPotential.__init__(self, resolution=resolution, range_=range_)
        self.debye_length = debye_length
        self.max_force = 200

    def potential(self, r, types):
        """ Electrostatics """
        typeA, typeB = types
        ld = self.debye_length
        q1 = typeA.charge
        q2 = typeB.charge
        D = 80                  # dielectric of water
        _z = 2.5
        D = 80 * (1- (r/_z)**2 * np.exp(r/_z)/(np.exp(r/_z)-1)**2)
        ## units "e**2 / (4 * pi * epsilon0 AA)" kcal_mol
        A =  332.06371
        u_elec = (A*q1*q2/D)*np.exp(-r/ld) / r

        """ LJ-type term """

        try:
            """ Rather than interacting through the hydrophobic
            potential phi_hp, cationic residues interact with aromatic
            residues through an 8–6 Lennard Jones potential """
            assert( all( t.version in ('1.0cp',1.1) for t in types ) )
            key = tuple((typeA.name[:3],typeB.name[:3]))
            eps = eps_cp_dict[key]

            _r_m = 4.5
            r6 = (_r_m/r)**6
            r8 = (_r_m/r)**8

            if ('is_FG' in typeA.__dict__ and 'is_BS' in typeB.__dict__) or \
               ('is_FG' in typeB.__dict__ and 'is_BS' in typeA.__dict__):
                raise           # don't use this path!
                eps = 3.2863289 # units "13.75 kJ" kcal

            logger.info(f'Writing potential with CP for {key = }, {eps = }')
            u_lj = eps * (3*r8 - 4*r6)
        except:

            alpha = 0.27
            epsilon_hp = 3.1070746 # units "13 kJ/N_A" kcal_mol
            epsilon_rep = 2.3900574 # units "10 kJ/N_A" kcal_mol

            if ('is_FG' in typeA.__dict__ and 'is_BS' in typeB.__dict__) or \
               ('is_FG' in typeB.__dict__ and 'is_BS' in typeA.__dict__):
                """ We assigned a specific interaction strength eps_BS,FG (rather than eps_ij) between binding site regions on Kap95 and any FG-motif residue """
                epsilon = 3.2863289 # units "13.75 kJ" kcal
                sigma = 6.0
            else:
                if 'nts' in typeA.__dict__ or 'nts' in typeB.__dict__:
                    epsilon = 0.1
                    sigma = 16
                else:
                    epsilon = epsilon_hp*np.sqrt( (typeA.epsilon*typeB.epsilon)**alpha )
                    sigma = 6.0
                    if ('is_folded' in typeA.__dict__ or 'is_folded' in typeB.__dict__):
                        epsilon = epsilon_rep

            r6 = (sigma/r)**6
            r8 = (sigma/r)**8

            u_lj = (epsilon_rep-epsilon) * r8
            s = r<=sigma
            u_lj[s] = epsilon_rep*r8[s] - epsilon*(4*r6[s]-1)/3
            u_lj = u_lj - u_lj[r>25][0]
            u_lj[r>25] = 0

        u = u_elec + u_lj
        return u

class OnckBeads(PolymerBeads):
    """
    A class that represents polymer beads in the Onck model.

    The Onck model is a coarse-grained model for proteins where each amino acid is represented
    by a single bead. The model includes specific bonded interactions (bonds, angles, dihedrals)
    based on the amino acid types in the sequence.

    Parameters
    ----------
    polymer : Polymer
        The parent polymer object to which these beads belong.
    sequence : list, optional
        The sequence of amino acid types for each bead, must match polymer length.
    spring_constant : float, default=38.422562
        Spring constant for peptide bonds in units of 8038 kJ/(N_A nm^2) or 0.5 kcal_mol/AA^2.
    rest_length : float, default=3.8
        Equilibrium length of peptide bonds.
    version : float, optional
        Version of the Onck model to use. Defaults to 1.1 if not specified.
    **kwargs
        Additional keyword arguments passed to the parent PolymerBeads constructor.
    types_dict : dict
        Mapping between amino acid codes and bead types.
    peptide_bond : HarmonicBond
        Bond potential used for peptide bonds between adjacent beads.

    Raises
    ------
    ValueError
        If the version is not supported or if the sequence length doesn't match the polymer length.
    NotImplementedError
        If sequence is None (random sequence generation not implemented).

    Note
    -----
    The Onck model distinguishes between proline (P), glycine (G), and all other amino acids (X)
    for certain bonded interactions. Special potentials are loaded from files for angle and 
    dihedral interactions based on this classification.
    """
    def __init__(self, polymer, sequence=None,
                 spring_constant = 38.422562, # units "8038 kJ / (N_A nm**2)" "0.5 * kcal_mol/AA**2"
                 rest_length=3.8, version=None, **kwargs):

        if version == None:
            logger.warning(f'No Onck model version specified; using version 1BPA-1.1 from 2023')
            version = 1.1
        if version not in _types_versions:
            raise ValueError(f'Unkown Onck model version "{version}"')
        self.version = version
        self.types_dict = _types_versions[version]
        _types = _types_versions[version] # Update global _types convenience variable

        self.peptide_bond = HarmonicBond(k = spring_constant,
                                         r0 = rest_length,
                                         range_ = (0,100),
                                         resolution = 0.01,
                                         max_force = 10)

        if sequence is None:
            raise NotImplementedError
            # ... set random sequence

        self.polymer = polymer
        self.sequence = sequence

        self.spring_constant = spring_constant
        PolymerBeads.__init__(self, polymer, sequence, rest_length=rest_length, **kwargs)

        assert(self.monomers_per_bead_group == 1)

        if len(sequence) != polymer.num_monomers:
            raise ValueError("Length of sequence does not match length of polymer")


    def _generate_ith_bead_group(self, i, r, o):
        s = self.sequence[i]
        return PointParticle(self.types_dict[s], r,
                             name = s,
                             resid = i+1)

    def _join_adjacent_bead_groups(self, ids):

        def bead_to_type(bead):
            if bead.type_.name == 'PRO':
                return 'P'
            elif bead.type_.name == 'GLY':
                return 'G'
            else:
                return 'X'

        ## Two consecutive groups
        if len(ids) == 2:
            b1,b2 = [self.children[i] for i in ids]
            """ units "10 kJ/N_A" kcal_mol """
            bond = self.peptide_bond
            self.add_bond( i=b1, j=b2, bond = bond, exclude=True )
        elif len(ids) == 3:
            b1,b2,b3 = [self.children[i] for i in ids]
            t1,t2,t3 = [bead_to_type(b) for b in (b1,b2,b3)]

            filename = 'onck_model_potentials/bend_O{}{}.txt'.format(
                t2, 'P' if t3 == 'P' else 'Y' )
            self.add_angle( i=b1, j=b2, k=b3,
                          angle = TabulatedPotential( get_resource_path(filename) ) )
            self.add_exclusion( i=b1, j=b3 )
        elif len(ids) == 4:
            ## Four consecutive monomers
            b1,b2,b3,b4 = [self.children[i] for i in ids]
            t1,t2,t3,t4 = [bead_to_type(b) for b in (b1,b2,b3,b4)]

            filename = 'onck_model_potentials/dih_{}{}.txt'.format(t2,t3)
            self.add_dihedral( i=b1, j=b2, k=b3, l=b4,
                               dihedral = TabulatedPotential( get_resource_path(filename) ) )
            self.add_exclusion( i=b1, j=b4 )
        else:
            raise Exception('Programming error!')

class OnckModel(PolymerModel):
    """
    A polymer model based on Onck's coarse-grained model for disordered FG-Nup peptides.

    The OnckModel class implements a coarse-grained model of polymers based on the work by Onck et al.
    This model is specifically designed for disordered FG-Nup peptides and incorporates
    electrostatic interactions through a Debye-Hückel potential.

    Parameters
    ----------
    polymers : list
        List of polymers to be modeled.
    sequences : list, optional
        List of amino acid sequences corresponding to each polymer.
        Must be provided. Default is None.
    debye_length : float, optional
        Debye screening length in angstroms. Default is 10.
    damping_coefficient : float, optional
        Damping coefficient in ns. Default is 100.
    version : float, optional
        Version of the Onck model to use. Default is 1.1 (corresponding to 1BPA-1.1 from 2023).
    DEBUG : bool, optional
        Whether to enable debug mode. Default is False.
    **kwargs : dict
        Additional keyword arguments to pass to the parent PolymerModel class.
        Common options include:
        - timestep: Simulation timestep (default: 20e-6)
        - cutoff: Cutoff distance for non-bonded interactions (default: max(5*debye_length, 25))
        - decomp_period: Decomposition period (default: 1000)
    types_dict : dict
        Dictionary mapping from type keys to type objects for the given version.
    types : list
        List of all types used in the model.

    Note
    -----
    The model differs from the published Onck model if a Debye length other than 12.7 Å is chosen.
    In such cases, the non-bonded cutoff is set to 5 * debye_length.

    References
    ----------
    Please refer to the citations displayed during initialization.
    """
    def __init__(self, polymers,
                 sequences = None,
                 debye_length = 10,
                 # damping_coefficient = 50e3,
                 damping_coefficient = 100,
                 version = None,
                 DEBUG=False,
                 **kwargs):

        """
        [debye_length]: angstroms
        [damping_coefficient]: 1/ns (zeta/m, written to .bd as transDamping)

        """

        if version == None:
            logger.warning(f'No Onck model version specified; using version 1BPA-1.1 from 2023')
            version = 1.1
        if version not in _types_versions:
            raise ValueError(f'Unkown Onck model version "{version}"')
        self.version = version
        self.types_dict = _types_versions[version]

        logger.info(f'Using an implementation of the Onck model {version_name[self.version]} for disordered FG-Nup peptides.')
        _msg = 'Please cite all appropriate articles, including:\n'
        _msg = _msg + "\n  and\n".join(ref.display() for ref in version_refs[self.version])
        print(_msg)

        if debye_length != 12.7:
            logger.warning("""Deviating from the model published by Onck by choosing a Debye length differing from 1.27 nm.
    Be advised that the non-bonded cutoff is simply set to 5 * debye_length, but this is not necessarily prescribed by the model.""")

        if 'timestep' not in kwargs: kwargs['timestep'] = 20e-6
        if 'cutoff' not in kwargs: kwargs['cutoff'] = max(5*debye_length,25)

        if 'decomp_period' not in kwargs:
            kwargs['decomp_period'] = 1000

        """ Assign sequences """
        if sequences is None:
            raise NotImplementedError("OnckModel must be provided a sequences argument")

        PolymerModel.__init__(self, polymers, sequences, monomers_per_bead_group=1, **kwargs)

        """ Update type diffusion coefficients """
        self.types = all_types = [t for key,t in self.types_dict.items()]
        self.set_damping_coefficient( damping_coefficient )

        """ Set up nonbonded interactions """
        self.nonbonded = nonbonded = OnckNonbonded(debye_length)
        for t in all_types:
            self._add_nonbonded_interaction(nonbonded, t)

    def _add_nonbonded_interaction(self, interaction, type_):
        i = self.types.index(type_) if type_ in self.types else 0
        for t in self.types[i:]:
            # self.add_nonbonded_interaction( interaction, typeA=type_, typeB=t )
            self.add_nonbonded_interaction( interaction, typeA=type_, typeB=t )

    def _generate_polymer_beads(self, polymer, sequence, polymer_index=None):
        return OnckBeads(polymer, sequence,
                         monomers_per_bead_group = self.monomers_per_bead_group,
                         poymer_index = polymer_index,
                         version = self.version
                         )

    def set_damping_coefficient(self, damping_coefficient):
        for t in self.types:
            t.damping_coefficient = damping_coefficient
            # t.diffusivity = 831447.2 * temperature / (t.mass * damping_coefficient)

if __name__ == "__main__":
    logger.info("TYPES")
    for n,t in _types.items():
        logger.info("{}\t{}\t{}\t{}\t{}".format(n, t.name, t.mass, t.charge, t.epsilon))
