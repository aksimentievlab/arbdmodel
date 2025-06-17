import numpy as np
# from scipy.spatial import distance_matrix
import parmed

from parmed import load_file
# from parmed.topologyobjects import Bond,Angle,Dihedral,Improper
# from parmed.charmm.parameters import CharmmParameterSet

# import MDAnalysis as mda
from arbdmodel import Group, ParticleType, PointParticle, ArbdModel, ArbdEngine, logger
from arbdmodel.interactions import HarmonicBond


""" Script for creating a dual topology

Caveats:

  1. Script is specialized such that dual topology atoms can _only_
     have bonded interactions within the residue they are associated

  2. Specialized for DNA, no CMAPs etc

"""

def read_files( psf, pdb, parameter_files=None, system_type = 'charmm'):
    if system_type == 'charmm':
        p1 = parmed.charmm.CharmmPsfFile(psf)
        c1 = parmed.load_file(pdb)
        for a,b in zip(p1.atoms,c1.atoms):
            a.xx = b.xx 
            a.xy = b.xy
            a.xz = b.xz
            a.bfactor = b.bfactor
        if parameter_files is not None:
            if isinstance(parameter_files,str):
                parameter_files = [parameter_files]
            # import ipdb; ipdb.set_trace()
            # params = parmed.charmm.CharmmParameterSet(parameter_files[-1])
            params = parmed.charmm.CharmmParameterSet(*parameter_files)
            p1.load_parameters( params )
    else:
        raise NotImplementedError(f'Cannot import a "{system_type}" model')
    return p1

def validate_atom(atom):
    """dict_keys(['list', '_idx', 'atomic_number', 'name', 'type', '_charge', 'mass', 'nb_idx', 'solvent_radius', 'screen', 'tree', 'join', 'irotat', 'bfactor', 'altloc', 'occupancy', '_bond_partners', '_angle_partners', '_dihedral_partners', '_tortor_partners', '_exclusion_partners', 'residue', 'marked', 'bonds', 'angles', 'dihedrals', 'urey_bradleys', 'impropers', 'cmaps', 'tortors', 'other_locations', 'atom_type', 'number', 'anisou', '_rmin', '_epsilon', '_rmin14', '_epsilon14', 'children', 'aromatic', 'formal_charge', 'hybridization', 'props', 'xx', 'xy', 'xz']) """

    try:
        atom.multipoles
        raise NotImplementedError(f'Atom {atom} uses multipoles attribute for AMEOBA')
    except AttributeError:
        pass

    for k in 'tortors other_locations anisou hybridization irotat tree screen solvent_radius join altloc marked cmaps children aromatic formal_charge'.split():
        _a = atom.__getattribute__(k)
        if _a is not None:
            _isnum = isinstance(_a,float) or isinstance(_a,int)
            if (_isnum and (_a != 0)) or ((not _isnum) and len(_a) > 0):
                _msg = f'Atom {atom.idx} "{k}" is non-null ({_a})'
                logger.warn(_msg)
                # raise NotImplementedError(_msg)

    
    if len(atom.other_locations) > 0:
        _msg = f'Other locations is not None {part} includes RB torsions'
        raise NotImplementedError(_msg)

    ...

def validate_topology(part):
    if len(part.rb_torsions) > 0:
        _msg = f'Topology {part} includes RB torsions'
        raise NotImplementedError(_msg)
    if len(part.urey_bradleys) > 0:
        _msg = f'Topology {part} includes Urey Bradley terms'
        raise NotImplementedError(_msg)
    if len(part.donors) > 0:
        _msg = f'Topology {part} includes donors'
        raise NotImplementedError(_msg)
    if len(part.acceptors) > 0:
        _msg = f'Topology {part} includes acceptors'
        raise NotImplementedError(_msg)
    if len(part.rb_torsions) > 0:
        _msg = f'Topology {part} includes RB torsions'
        raise NotImplementedError(_msg)

    for k in ['impropers', 'cmaps',
              'trigonal_angles',
              'out_of_plane_bends',
              'pi_torsions',
              'stretch_bends',
              'torsion_torsions',
              'chiral_frames',
              'multipole_frames',
              'adjusts',
              'links',
              ]:
        _a = part.__getattribute__(k)
        # print(k)
        if len(_a) > 0:
            _msg = f'Topology {part} includes {len(_a)} "{k}"'
            raise NotImplementedError(_msg)

    if part.unknown_functional != False:
        _msg = f'Topology {part} unknown_functional is "{part.unknown_functional}" (not "False")'
        raise NotImplementedError(_msg)

    if part.symmetry is not None:
        _msg = f'Topology {part} symmetry is "{part.symmetry}" (not "None")'
        raise NotImplementedError(_msg)

    if part.nrexcl > 4:
        _msg = f'Topology {part} nrexcl is "{part.nrexcl}" (not <= "4")'
        raise NotImplementedError(_msg)
        
    for k in ['groups','space_group']:
        if len(_a) > 0:
            _msg = f'Topology {part} includes {len(_a)} "{k}" that are not converted in any way'
            logger.warn(_msg)
        
    if p._combining_rule != 'lorentz':
        _msg = f'Unrecognized non-bonded combining rule "{p._combining_rule}"'
        raise NotImplementedError(_msg)
               ## '_box', '_coordinates', 'space_group', 'unknown_functional', 'nrexcl', 'title', '_combining_rule', 'symmetry', 'name', 'flags'])

    validate_atom(p.atoms[0])
        

pre='2-seq001.TTAGCCGA/1-build/output/single'
ppre = '/data/server3/cmaffeo2/systems/2023-HJ/1-protocol/AMBERff-in-NAMD/OL15'
# p = read_files( f'{pre}.psf', f'{pre}.pdb', [f'{ppre}/parm10.prm', f'{ppre}/frcmod.DNA.OL15.prm', f'{"/".join(pre.split("/")[:2])}/par_water_ions_cufix-3.prm'] )
# p = read_files( f'{pre}.psf', f'{pre}.pdb', [f'{ppre}/parm10.prm', f'{ppre}/frcmod.DNA.OL15.prm', 'par_water_ions_cufix-3.prm'] )
p = read_files( f'{pre}.psf', f'{pre}.pdb', [f'{"/".join(pre.split("/")[:2])}/parm10.prm', f'{ppre}/frcmod.DNA.OL15.prm', 'tip3p_ions.str', 'par_water_ions_cufix-3.prm'] )

logger.info('Importing from ParmEd types')
_atypes = set(a.atom_type for a in p.atoms)
logger.info(f'Importing {len(_atypes)} types from ParmEd')
_new_types_d = {}
for t in _atypes:
    _kwargs = { k:t.__getattribute__(k)
                for k in 'number mass atomic_number epsilon rmin charge'.split() }

    for k in 'epsilon_14 rmin_14'.split():
        _a1 = t.__getattribute__(k)
        _a2 = t.__getattribute__(k.split('_')[0])
        if _a1 != _a2:
            _msg = f'{t}.{k} differs from normal term ({_a1} != {_a2})'
            logger.warn(_msg)
            # raise NotImplementedError(_msg)

    for k in 'nbfix nbthole _bond_type'.split():
        _a = t.__getattribute__(k)
        if _a is not None and len(_a) > 0:
            _msg = f'{t}.{k} is non-null ({_a})'
            logger.warn(_msg)
            # raise NotImplementedError(_msg)

    assert( t.name not in _new_types_d)
    _new_types_d[t.name] = ParticleType(t.name,
                                        damping_coefficient=1e-4, # 0.1 ps
                                        **_kwargs)

allsegs = Group(name='allsegs', _segments={})
atoms_d = {}
bond_t_d = {}
angle_t_d = {}
dihed_t_d = {}

def _get_residue(res):
    segid = res.segid if res.segid is not None else res.chain
    if segid not in allsegs._segments:
        allsegs._segments[segid] = Group(name=segid, _residues={}, parent=allsegs)
        logger.info(f'Creating segment {segid}')
    seg = allsegs._segments[segid]

    key = (segid, res.name, res.number)
    if key not in seg._residues:
        seg._residues[key] = Group(name=res.name, resid=res.number, chain=res.chain, parent=seg)
        logger.debug((seg, segid, res.name, res.number, res._idx))

    return seg._residues[key]

def _add_atom(atom):
    a = atom
    # print( a.__dict__.keys() )
    mapping = dict(bfactor='beta',
                   occupancy=None,
                   )
    for k,v in list(mapping.items()):
        if v is None: mapping[k] = k

    res = _get_residue(a.residue)
    # if res not in _add_residue
    # if seg not in segs:
    #     _add_segment(seg)

    _a = PointParticle(name = a.name, type_=_new_types_d[a.type],
                       position=np.array((a.xx,a.xy,a.xz)), parent=res,
                       **{mapping[k]:a.__getattribute__(k) for k in mapping.keys()}) # get attributes

    atoms_d[a] = _a

def _get_bond_type(bond):
    t = bond.type
    if t not in bond_t_d:
        if t.penalty is not None:
            raise NotImplementedError(f'Bond.penalty for bond {t} not null')
        width = 0.76565392/np.sqrt(np.abs(t.k)) # units "sqrt(295 k K/ (1 kcal_mol/AA**2))" AA
        b = HarmonicBond( k = t.k, r0 = t.req,
                          resolution = 0.02*width,
                          range_=[max(0,t.req-5*width),t.req+5*width] )
        bond_t_d[t] = b
    return bond_t_d[t]

logger.info(f'Importing {len(p.atoms)} atoms from ParmEd')
for i,a in enumerate(p.atoms):
    assert( a.idx == i )
    _add_atom(a)
    # if i > 50: break

logger.info(f'Importing {len(p.bonds)} bonds from ParmEd')
for b0 in p.bonds:
    t = _get_bond_type(b0)
    a1,a2 = [atoms_d[a] for a in (b0.atom1, b0.atom2)]
    seg = a1.parent.parent
    seg.add_bond( a1, a2, t, p.nrexcl >= 1 )
    
model = ArbdModel( [allsegs], integrator = 'MD', cutoff=12 )

engine = ArbdEngine( timestep=1e-6 )

engine.simulate(model,'run', directory='test', timestep = 1e-7,
                num_steps=10000, output_period=10)

    
# print(atoms)
# print(p.atoms)
# validate_topology(p)
# validate_atom(p.atoms[0])

