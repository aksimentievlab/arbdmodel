from chrispy import logger
from pathlib import Path
import dill

import numpy as np
from mrrna import DoubleStrandedSegment
from mrrna.arbdmodel.coords import rotationAboutAxis, quaternion_to_matrix, quaternion_from_matrix, quaternion_slerp, readArbdCoords
from mrrna.arbdmodel import Group, ArbdModel

from mrrna.readers.segmentmodel_from_lists import model_from_basepair_stack_3prime

from shape_cg import ShapeCGNonbonded, ShapeCGFactory

from project_data import pdb_list, get_copy_number


dry_run = True

_pdb_name = {k:k[0]+k[-2:] for k in pdb_list}
assert( len(set([v for k,v in _pdb_name.items()])) == len(pdb_list) )

protein_factory = {k:ShapeCGFactory(f'{k}_autopsf.psf',
                                    f'{k}_autopsf.pdb',
                                    name=_pdb_name[k])
                   for k in pdb_list}

box_size = 2*(200+50)
    
_total_mass_100000 = sum([protein_factory[k]._fine_total_mass * get_copy_number(k,100000) for k in pdb_list])
# units "200 mg/ml" "dalton / (1200 AA)**3"
#   * 2.0812523e+08
rng = np.random.default_rng(seed=12345)

print(_total_mass_100000)

def __add_types_to_protein_factory_dict(protein_counts, num_CG_sites=None):
    prot_types = []
    for k in pdb_list:
        if protein_counts[k] == 0: continue
        prot_types.extend(protein_factory[k].get_coarse_types(num_CG_sites))
    protein_factory[('__types__',num_CG_sites)] = prot_types

def concentration_to_debye_length(c, temperature=295):
    """ c given in mM """
    # sqrt( 80 epsilon0 295 k K / ((150 mM) e**2/particle) )
    return 11.154259 * np.sqrt((150/c)*(295/temperature))
    
def __add_protein_nb_interactions(model, prot_types, nt_sigma=10, debye_length = None):
    if debye_length is None:
        # model.nbSchemes[0][0].debye_length
        debye_length = concentration_to_debye_length(150)

    prot_nb = ShapeCGNonbonded(debye_length = debye_length)
    old_interactions = model.nbSchemes
    model.nbSchemes = []

    for i,prot_t1 in enumerate( prot_types ):
        for j,prot_t2 in enumerate( prot_types[i:],i ):
            model.useNonbondedScheme( prot_nb, typeA=prot_t1, typeB=prot_t2 )
        try:
            for t in model.nucleic_types:
                if t.name[0] == 'O': continue
                t.charge = -1 * t.nts
                t.sigma = nt_sigma            #
                model.useNonbondedScheme( prot_nb, typeA=t, typeB=prot_t1 )
        except:
            pass
    model.nbSchemes = model.nbSchemes + old_interactions

def __add_proteins_to_model(model, protein_counts, protein_positions, num_CG_sites=None, nt_sigma=10):
    all_pro = []
    start = 0
    for k in pdb_list:
        _n = protein_counts[k]
        if _n == 0: continue
        pro_parts = [protein_factory[k].generate_protein(protein_positions[start+i], index=i, num_CG_sites=num_CG_sites)
                     for i in range(_n)]
        g = Group(name=f'all_{k}', children = pro_parts)
        all_pro.append(g)
        start += _n

    protein_group = Group(name='cytosol', children=all_pro)
    model.add( protein_group )
    key = ('__types__',num_CG_sites)
    if key not in protein_factory: __add_types_to_protein_factory_dict(protein_counts, num_CG_sites)
    __add_protein_nb_interactions(model, protein_factory[key], nt_sigma=nt_sigma)
    return protein_group

def run(gpu, protein_density, num_CG_sites=None, concentration=150):

    radius=200
    debye_length=concentration_to_debye_length(150)
    # units "200 mg/ml" "dalton / (1200 AA)**3"
    #   * 2.0812523e+08
    total_proteins_target = (protein_density/200) * (2.081e8) * ((4*np.pi/3) * radius**3/1200**3) * 100000 / _total_mass_100000
    print(total_proteins_target)
    protein_counts = {k: get_copy_number(k, total_proteins_target) for k in pdb_list}
    total_proteins = sum([v for k,v in protein_counts.items()])

    _x,_y,_z = np.eye(3)
    print(_z)
    
    angles = np.random.random( (total_proteins, 2) )
    angles[:,0] = (angles[:,0])*360
    angles[:,1] = np.arccos(1-2*angles[:,1])*180/np.pi

    pro_rots = [rotationAboutAxis(_z,a0).dot(rotationAboutAxis(_x,a1))
                for a0,a1 in angles]
    radii = np.random.random( (total_proteins, 1 ) )**(1/3) * radius
    pro_centers = [R[:,2]*r for R,r in zip(pro_rots,radii)]

    print(f'total_proteins: {total_proteins}')

    name = f'rho_{protein_density}_{concentration}'
    directory = f'run_{num_CG_sites}'
    outfile = f'{directory}/{name}.bd'

    model = ArbdModel([], cutoff=200, dimensions = [box_size]*3)
    nbSchemes0 = list(model.nbSchemes)

    restart_file = f'run_1/output/min_{protein_density}.restart'
    if not Path(restart_file).exists():

        protein_group = __add_proteins_to_model(model, protein_counts, pro_centers, num_CG_sites=1)
        print( set([p.type_ for p in protein_group]) )
        for t in [p.type_ for p in protein_group]:
            try:
                t.modified
            except:
                t.modified = True
                t.sigma = 2*t.sigma
                t.grid = [(f'../confine-{radius}.dx',1)]

        for s,t1,t2 in model.nbSchemes:
            try:
                t1.modified
            except:
                t1.modified = True
                t1.sigma = 2*t1.sigma
                t1.grid = [(f'../confine-{radius}.dx',1)]
            try:
                t2.modified
            except:
                t2.modified = True
                t2.sigma = 2*t2.sigma
                t2.grid = [(f'../confine-{radius}.dx',1)]

        model.simulate( f'min_{protein_density}', directory='run_1',
                         num_steps = 1e5,
                         output_period = 1e4,
                         timestep = 200e-6,
                         gpu = gpu,
                         # restart_file = f'output/{oldname}.restart' if oldname is not None else None,
                         dry_run=False
                        )

        ## Restore model
        model.remove(protein_group)
        for p in model:
            p.restraints = []
        model.nbSchemes = nbSchemes0

    new_protein_coords = readArbdCoords(restart_file)[len(model):]

    protein_group = __add_proteins_to_model(model, protein_counts, new_protein_coords, num_CG_sites=num_CG_sites)
    for t in [p.type_ for p in protein_group]:
        t.grid = [(f'../confine-{radius}.dx',1)]

    for s,t1,t2 in model.nbSchemes:
        t1.grid =  [(f'../confine-{radius}.dx',1)]
        if t1 is not t2:
            t2.grid =  [(f'../confine-{radius}.dx',1)]


    model.simulate( name, directory=directory,
                    num_steps = 1e8,
                    output_period = 1e4,
                    timestep = 200e-6,
                    gpu = gpu,
                    # restart_file = f'output/{oldname}.restart' if oldname is not None else None,
                    dry_run=dry_run
        )

if __name__ == '__main__':
    gpu=0
    # for density in [19.5,44.6,55.8,84.5,101.1,200,400]:
    conc=150
    # conc=1000
    density0 =  [101.1,200,400]
    for density in [55.8,84.5,101.1,200,400]:
        # if density in density0: continue
    # for density in [84.5,101.1,200,400]:
    # for density in [101.1,200,400]:
        # for N_beads in (1,2,4,8,16,32,64):
        # for N_beads in (4,8,16):
        for N_beads in (2,32,64):
            run(gpu, density, N_beads, concentration=conc)
