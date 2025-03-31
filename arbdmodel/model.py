import numpy as np
from copy import copy, deepcopy
from glob import glob
import MDAnalysis as mda

from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm

from .logger import devlogger, logger, get_resource_path
from .interactions import NullPotential
from . import Transformable, Parent, Group
from .sim_config import SimConf
from .core_objects import GroupSite
from .engine import ArbdEngine, NamdEngine
from .coords import calculate_dimensions_from_cell_vectors


class PdbModel(Transformable, Parent):

    def __init__(self, children=None, dimensions=None, remove_duplicate_bonded_terms=False):
        Transformable.__init__(self,(0,0,0))
        Parent.__init__(self, children, remove_duplicate_bonded_terms)
        self.dimensions = dimensions
        self.particles = [p for p in self if not p.rigid]
        self.rigid_bodies = [p for p in self if p.rigid]
        self.cacheInvalid = True

    def _updateParticleOrder(self):
        pass

    def write_pdb(self, filename, beta_from_fixed=False):
        if self.cacheInvalid:
            self._updateParticleOrder()
        with open(filename,'w') as fh:
            ## Write header
            fh.write("CRYST1{:>9.3f}{:>9.3f}{:>9.3f}  90.00  90.00  90.00 P 1           1\n".format( *self.dimensions ))

            ## Write coordinates
            formatString = "ATOM {idx:>6.6s} {name:^4.4s} {resname:3.3s} {chain:1.1s}{resid:>5.5s}   {x:8.8s}{y:8.8s}{z:8.8s}{occupancy:6.2f}{beta:6.2f}  {charge:2d}{segname:>6s}\n"
            for p in self.particles:
                ## http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
                data = p._get_psfpdb_dictionary()
                idx = data['idx']

                if np.log10(idx) >= 5:
                    idx = " *****"
                else:
                    idx = "{:>6d}".format(idx)
                data['idx'] = idx

                if beta_from_fixed:
                    data['beta'] = 1 if 'fixed' in p.__dict__ else 0

                pos = p.get_collapsed_position()
                dig = [max(int(np.log10(np.abs(x)+1e-6)//1),0)+1 for x in pos]
                for d in dig: assert( d <= 7 )
                # assert( np.all(dig <= 7) )
                fs = ["{: %d.%df}" % (8,7-d) for d in dig]
                x,y,z = [f.format(x) for f,x in zip(fs,pos)] 
                data['x'] = x
                data['y'] = y
                data['z'] = z
                assert(data['resid'] < 1e5)
                data['charge'] = int(data['charge'])
                data['resid'] = "{:<4d}".format(data['resid'])
                fh.write( formatString.format(**data) )

        return

    def write_pqr(self, filename):
        if self.cacheInvalid:
            self._updateParticleOrder()
        with open(filename,'w') as fh:
            ## Write header
            fh.write("CRYST1{:>9.3f}{:>9.3f}{:>9.3f}  90.00  90.00  90.00 P 1           1\n".format( *self.dimensions ))

            ## Write coordinates
            formatString = "ATOM {idx:>6.6s} {name:^4.4s} {resname:3.3s} {chain:1.1s} {resid:>5.5s}   {x:.6f} {y:.6f} {z:.6f} {charge} {radius}\n"
            for p in self.particles:
                data = p._get_psfpdb_dictionary()

                idx = data['idx']
                if np.log10(idx) >= 5:
                    idx = " *****"
                else:
                    idx = "{:>6d}".format(idx)
                data['idx'] = idx

                x,y,z = p.get_collapsed_position()
                data['x'] = x
                data['y'] = y
                data['z'] = z
                assert(data['resid'] < 1e5)
                data['resid'] = "{:<4d}".format(data['resid'])
                if 'radius' not in data:
                    data['radius'] = 2 * (data['mass']/16)**0.333333
                fh.write( formatString.format(**data) )
        return
        
    def write_psf(self, filename):
        if self.cacheUpToDate == False:
            self._updateParticleOrder()
        with open(filename,'w') as fh:
            ## Write header
            fh.write("PSF NAMD\n\n") # create NAMD formatted psf
            fh.write("{:>8d} !NTITLE\n\n".format(0))
            
            ## ATOMS section
            fh.write("{:>8d} !NATOM\n".format(len(self.particles)))

            ## From vmd/plugins/molfile_plugin/src/psfplugin.c
            ## "%d %7s %10s %7s %7s %7s %f %f"
            formatString = "{idx:>8d} {segname:7.7s} {resid:<10.10s} {resname:7.7s}" + \
                           " {name:7.7s} {type:7.7s} {charge:f} {mass:f}\n"
            for p in self.particles:
                data = p._get_psfpdb_dictionary()
                data['resid'] = "%d%c%c" % (data['resid']," "," ") # TODO: work with large indices
                fh.write( formatString.format(**data) )
            fh.write("\n")

            ## Write out bonds
            bonds = self.get_bonds()
            fh.write("{:>8d} !NBOND\n".format(len(bonds)))
            counter = 0
            for p1,p2,b,ex in bonds:
                fh.write( "{:>8d}{:>8d}".format(p1.idx+1,p2.idx+1) )
                counter += 1
                if counter == 4:
                    fh.write("\n")
                    counter = 0
                else:
                    fh.write(" ")
            fh.write("\n" if counter == 0 else "\n\n")

            ## Write out angles
            angles = self.get_angles()
            fh.write("{:>8d} !NTHETA\n".format(len(angles)))
            counter = 0
            for p1,p2,p3,a in angles:
                fh.write( "{:>8d}{:>8d}{:>8d}".format(p1.idx+1,p2.idx+1,p3.idx+1) )
                counter += 1
                if counter == 3:
                    fh.write("\n")
                    counter = 0
                else:
                    fh.write(" ")
            fh.write("\n" if counter == 0 else "\n\n")

            ## Write out dihedrals
            dihedrals = self.get_dihedrals()
            fh.write("{:>8d} !NPHI\n".format(len(dihedrals)))
            counter = 0
            for p1,p2,p3,p4,a in dihedrals:
                fh.write( "{:>8d}{:>8d}{:>8d}{:>8d}".format(p1.idx+1,p2.idx+1,p3.idx+1,p4.idx+1) )
                counter += 1
                if counter == 2:
                    fh.write("\n")
                    counter = 0
                else:
                    fh.write(" ") 
            fh.write("\n" if counter == 0 else "\n\n")

            ## Write out impropers
            impropers = self.get_impropers()
            fh.write("{:>8d} !NIMPHI\n".format(len(impropers)))
            counter = 0
            for p1,p2,p3,p4,a in impropers:
                fh.write( "{:>8d}{:>8d}{:>8d}{:>8d}".format(p1.idx+1,p2.idx+1,p3.idx+1,p4.idx+1) )
                counter += 1
                if counter == 2:
                    fh.write("\n")
                    counter = 0
                else:
                    fh.write(" ")
            fh.write("\n" if counter == 0 else "\n\n")

            fh.write("\n{:>8d} !NDON: donors\n\n\n".format(0))
            fh.write("\n{:>8d} !NACC: acceptors\n\n\n".format(0))
            fh.write("\n       0 !NNB\n\n")
            natoms = len(self.particles)
            for i in range(natoms//8):
                fh.write("      0       0       0       0       0       0       0       0\n")
            for i in range(natoms-8*(natoms//8)):
                fh.write("      0")
            fh.write("\n\n       1       0 !NGRP\n\n")


class ArbdModel(PdbModel):
    """
    Advanced model class for ARBD simulations with improved cell vector handling.
    """
    def __init__(self, children, cell_vectors=None, cell_origin=None, dimensions=None,
                 remove_duplicate_bonded_terms=True, buffer_factor=1.2,
                 configuration=None, dummy_types=tuple(), **conf_params):
        """
        Initialize ArbdModel with improved cell vector and dimension handling.
        
        Args:
            children: Child objects for the model
            cell_vectors: List of 3 cell basis vectors [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]]
            cell_origin: Cell origin coordinates [x,y,z]
            dimensions: Explicit dimensions for the simulation box (overrides cell_vectors)
            remove_duplicate_bonded_terms: Whether to remove duplicate bonded terms
            buffer_factor: Factor to scale dimensions derived from cell vectors
            configuration: SimConf object with configuration parameters
            dummy_types: Deprecated parameter, kept for backward compatibility
            **conf_params: Additional configuration parameters
        """
        # Calculate dimensions from cell vectors if provided and dimensions not explicitly set
        if dimensions is None and cell_vectors is not None:
            dimensions = calculate_dimensions_from_cell_vectors(
                cell_vectors, cell_origin, buffer_factor)
        
        # Default dimensions if neither dimensions nor cell vectors provided
        if dimensions is None:
            dimensions = (1000, 1000, 1000)
            
        # Set default cell vectors if not provided
        if cell_vectors is None:
            cell_vectors = [
                [dimensions[0], 0, 0],
                [0, dimensions[1], 0],
                [0, 0, dimensions[2]]
            ]
            
        # Set default cell origin if not provided
        if cell_origin is None:
            cell_origin = [0, 0, 0]
            
        # Initialize parent class
        PdbModel.__init__(self, children, dimensions, remove_duplicate_bonded_terms)
        
        
        # Store origin which might be different from cell_origin
        self.origin = cell_origin

        # Set up configuration
        if configuration is None: 
            configuration = SimConf(**conf_params)
        self.configuration = configuration

        # Initialize model properties
        self.num_particles = 0
        self.particles = []  
        self.type_counts = None 

        self.dummy_types = dummy_types  # TODO, determine whether these are really needed
        if len(self.dummy_types) != 0:
            raise("Dummy types have been deprecated")
        
        self.nonbonded_interactions = []
        self._nonbonded_interaction_files = []  # This could be made more robust

        self.cacheUpToDate = False

        self.group_sites = []

    def add(self, obj):
        self._clear_types()     # TODO: call upon (potentially nested) `child.add(add)`
        return Parent.add(self, obj)

    def add_group_site(self, particles, weights=None):
        g = GroupSite(particles, weights)
        self.group_sites.append(g)
        return g

    def _clear_types(self):
        devlogger.debug(f'{self}: Clearing types') 
        self.type_counts = None
        
    
    def clear_all(self, keep_children=False):
        Parent.clear_all(self, keep_children=keep_children)
        self.particles = []
        self.num_particles = 0
        self._clear_types()
        self.group_sites = []
        self._nonbonded_interaction_files = []

    def extend(self, other_model, copy=False):

        if any( [p.rigid for p in self] + [p.rigid for p in other_model] ):
            raise NotImplementedError('Models containing rigid bodies cannot yet be extended')
        
        assert( isinstance(other_model, ArbdModel) )
        if copy == True:
            logger.warning(f'Forcing {self.__class__}.extend(other_model,copy=False)')
            copy = False

        ## Combine particle types, taking care to handle name clashes
        self._countParticleTypes()
        other_model._countParticleTypes()

        self.getParticleTypesAndCounts()
        other_model.getParticleTypesAndCounts()

        names = set([t.name for t in self.type_counts.keys()])

        devlogger.debug(f'Combining types {self.type_counts.keys()} and {other_model.type_counts.keys()}')
        for t1 in self.type_counts.keys():
            for t2 in other_model.type_counts.keys():
                if t1.name == t2.name and t1.__eq__(t2, check_equal=False):
                    i = 1
                    new_name = f'{t2.name}{i}'
                    while new_name in names:
                        i += 1
                        new_name = f'{t2.name}{i}'
                    devlogger.debug(f'Updating {t2.name} to {new_name}')
                    t2.name = new_name
                    names.add(new_name)
                    
        ## Combine interactions
        for i, tA, tB in other_model.nonbonded_interactions:
            devlogger.debug(f'Combining model interactions {i} {tA} {tB}')
            self.add_nonbonded_interaction(i,tA,tB)
                    
        # for g in other_model.children:
        #     self.update(g, copy=copy)
        g = Group()
        for attr in 'children position orientation bonds angles dihedrals impropers exclusions vector_angles bond_angles product_potentials group_sites'.split():
            g.__setattr__(attr, other_model.__getattribute__(attr))
        devlogger.debug(f'Updating {self} with {g.children[0].children[0]}')
        self.update(g, copy=copy)

        self._clear_types()
        
        ## Combine configurations
        self.configuration = other_model.configuration.combine(self.configuration, policy='best', warn=True)

    def assign_IBI_degrees_of_freedom(self):

        """ Convenience routine that adds degrees of freedom to
        corresponding IBI potentials """

        from .ibi import BondDof, AngleDof, DihedralDof, PairDistributionDof

        self.bonded_ibi_potentials = set()
        self.nonbonded_ibi_potentials = set()

        for get_fn, parts_pot_fn, cls in (
                (self.get_bonds,     lambda x: (x[:2],x[2]), BondDof),
                (self.get_angles,    lambda x: (x[:3],x[3]), AngleDof),
                (self.get_dihedrals, lambda x: (x[:4],x[4]), DihedralDof)
        ):
            for x in get_fn():
                parts, pot = parts_pot_fn(x)
                try: pot.degrees_of_freedom
                except: continue
                pot.degrees_of_freedom.append( cls(*parts) )
                if pot not in self.bonded_ibi_potentials:
                    self.bonded_ibi_potentials.add( pot )

        logger.info(f'Gathering nonbonded IBI degrees of freedom')
        all_exclusions = self.get_exclusions()
        try:
            raise               # particle.idx wasn't set without this
            if len(self.type_counts) == 0: raise
        except:
            self.getParticleTypesAndCounts()

        type_to_particles = dict()
        for p in self:
            t = p.type_
            if t not in type_to_particles: type_to_particles[t] = []
            type_to_particles[t].append(p)
        type_to_particles[None] = [p for p in self]

        already_handled = set()
        for pot, tA, tB in self.nonbonded_interactions:
            ## Avoid including particle type pairs that don't apply
            if (tA,tB) in already_handled: continue
            already_handled.add((tA,tB))
            already_handled.add((tB,tA))

            try:
                pot.degrees_of_freedom
            except:
                continue

            def _cond(pair):
                b1 = (tA is None or pair[0].type_ == tA) and (tB is None or pair[1].type_ == tB)
                b2 = (tB is None or pair[0].type_ == tB) and (tA is None or pair[1].type_ == tA)
                return (b1 or b2)

            ex = list(filter(_cond, all_exclusions))
            dof = PairDistributionDof( type_to_particles[tA], type_to_particles[tB], exclusions=ex,
                                       range_=pot.range_, resolution=pot.resolution)

            pot.degrees_of_freedom.append( dof )

            if pot not in self.nonbonded_ibi_potentials:
                self.nonbonded_ibi_potentials.add( pot )

    def load_target_IBI_distributions(self):
        raise NotImplementedError

    def run_IBI(self, iterations, directory = './', engine = None, replicas = 1, run_minimization = True, first_iteration=1, target_universe = None, target_trajectory = None):
        """
        Run Iterative Boltzmann Inversion (IBI) to optimize coarse-grained potentials.
        
        This method performs IBI iterations to match target distributions. For each iteration,
        it writes CG potentials, runs simulations, and extracts distributions from the resulting
        trajectories to update the potentials for the next iteration.
        
        Parameters
        ----------
        iterations : int
            Number of IBI iterations to run
        directory : str, optional
            Directory to store output files, defaults to current directory
        engine : ArbdEngine, optional
            Simulation engine to use. If None, creates a default engine with 5e6 steps
            and 1e4 output period
        replicas : int, optional
            Number of replica simulations to run per iteration, defaults to 1
        run_minimization : bool, optional
            Whether to run a brief minimization before the first iteration, defaults to True
        first_iteration : int, optional
            Starting iteration number, useful for continuing previous IBI runs, defaults to 1
        target_universe : MDAnalysis.Universe, optional
            Universe object containing target structure and trajectory
        target_trajectory : str or list, optional
            Target trajectory file(s) to use if target_universe is provided
            
        Raises
        ------
        ValueError
            If the model does not have any IBI potentials assigned
        NotImplementedError
            If the system does not have an orthorhombic unit cell
            
        Notes
        -----
        The method requires bonded_ibi_potentials and nonbonded_ibi_potentials to be 
        defined in the model. Use assign_IBI_degrees_of_freedom() before calling this method.
        """
        try:
            assert( len(self.bonded_ibi_potentials) + len(self.nonbonded_ibi_potentials) > 0 )
        except:
            raise ValueError('Model does not appear to contain IBI potentials; perhaps you forgot to run self.assign_IBI_degrees_of_freedom()')
        logger.info(f'Running {iterations} IBI iterations with {len(self.bonded_ibi_potentials)} bonded and {len(self.nonbonded_ibi_potentials)} nonbonded IBI potentials')

        if engine is None:
            engine = ArbdEngine(
                num_steps = 5e6,
                output_period = 1e4,
            )

        if np.array(self.dimensions).size > 3:
            raise NotImplementedError('IBI only implemented for systems with orthorhombic unit cells')
        else:
            box = tuple(list(self.dimensions[:3]) + ([90]*3))

        restart_file = None

        if target_universe is not None:
            _potentials = list(self.bonded_ibi_potentials)+list(self.nonbonded_ibi_potentials)
            with logging_redirect_tqdm(loggers=[logger,devlogger]):
                for potential in tqdm(_potentials, desc='Calculating target distributions'):
                    potential.get_target_distribution(target_universe, trajectory=target_trajectory)

        def _load_cg_u(iteration):
            name = f'ibi-{iteration:03d}'
            psf = '{}/{}.psf'.format(directory,name)
            globstring=f'{directory}/output/{name}.*dcd'
            dcds = [f for f in glob(globstring) if 'momentum' not in f]
            if len(dcds) == 0: raise ValueError(f'Expected to find dcds at {globstring}')
            cg_u = mda.Universe(psf,*dcds)
            return cg_u

        cg_u = None
        if first_iteration > 1:
            cg_u = _load_cg_u(first_iteration-1)
            with logging_redirect_tqdm(loggers=[logger,devlogger]):
                for p in tqdm(_potentials, desc='Calculating initial CG distributions'):
                    p.get_cg_distribution(cg_u, box=box, recalculate=False)

        for p in _potentials:
            p.iteration = first_iteration

        for i in range(first_iteration, iterations+1):
            logger.info(f'Working on IBI iteration {i}/{iterations}')

            with logging_redirect_tqdm(loggers=[logger,devlogger]):
                for p in tqdm(_potentials, desc='Writing CG potentials'):
                    # if 'IBIPotentials/intrabond-1' in p.filename(): import ipdb; ipdb.set_trace()
                    try:    p.write_cg_potential(cg_u, tol=p.tol, box=box)
                    except: p.write_cg_potential(cg_u, box=box)

            if i == 1 and run_minimization:
                logger.info(f'Running brief simulation with small timestep')
                ts0 = engine._get_combined_conf(self).timestep
                engine.simulate( self,
                                 output_name = 'ibi-min', directory = directory,
                                 timestep = ts0/100,
                                 num_steps = 10000, output_period=1000 )

                restart_file = f'{directory}/output/ibi-min.restart'

            name = f'ibi-{i:03d}'
            engine.simulate( self,
                             output_name = name, directory = directory,
                             restart_file = restart_file,
                             replicas = replicas )

            restart_file = f'{directory}/output/{name}{".0" if replicas > 1 else ""}.restart'
            cg_u = _load_cg_u(i)

            with logging_redirect_tqdm(loggers=[logger,devlogger]):
                for p in tqdm(_potentials, desc='Extracting CG distributions'):
                    p.get_cg_distribution(cg_u, box=box, recalculate=False)
                    p.iteration += 1
        
    def update(self, group , copy=False):
        assert( isinstance(group, Group) )
        if copy:
            group = deepcopy(group)
        group.parent = self
        self.add(group)
        
    def _get_nonbonded_interaction(self, typeA, typeB):
        for cp in (False,True):
            for s,A,B in self.nonbonded_interactions:
                if A is None or B is None:
                    if A is None and B is None:
                        return s
                    elif A is None and typeB.is_same_type(B, consider_parents=cp):
                        return s
                    elif B is None and typeA.is_same_type(A, consider_parents=cp):
                        return s
                elif typeA.is_same_type(A,consider_parents=False) and typeB.is_same_type(B,consider_parents=cp):
                    return s
        
        # raise Exception("No nonbonded scheme found for %s and %s" % (typeA.name, typeB.name))
        # print("WARNING: No nonbonded scheme found for %s and %s" % (typeA.name, typeB.name))

    def _countParticleTypes(self):
        ## TODO: check for modifications to particle that require
        ## automatic generation of new particle type
        type_counts = dict()    # type is key, value is 2-element list of regular particle counts and attached particles

        parts, self.rigid_bodies = [],[]
        for p in self:
            if p.rigid:
                self.rigid_bodies.append(p)
            else:
                parts.append(p)

        for p in parts:
            t = p.type_
            if t in type_counts:
                type_counts[t][0] += 1
            else:
                type_counts[t] = [1,0]

        parts = [p for rb in self if rb.rigid for p in rb.attached_particles]
        for p in parts:
            t = p.type_
            if t in type_counts:
                type_counts[t][1] += 1
            else:
                type_counts[t] = [0,1]
        
        if len(self.dummy_types) != 0:
            raise("Dummy types have been deprecated")
        # for t in self.dummy_types:
        #     if t not in type_counts:
        #         type_counts[t] = 0

        for i,tA,tB in self.nonbonded_interactions:
            if tA is not None and tA not in type_counts:
                type_counts[tA] = [0,0]
            if tB is not None and tB not in type_counts:
                type_counts[tB] = [0,0]


        self.type_counts = type_counts

        rbtc = dict()
        rbti={}
        for rb in self.rigid_bodies:
            t = rb.type_.name
            if t in rbtc: rbtc[t] += 1
            else:
                rbti[t]=rb.type_  
                rbtc[t] = 1
        self.rigid_body_type_counts = [(k,rbtc[k]) for k in sorted(rbtc.keys())]
        self.rigid_body_index=rbti
        devlogger.debug(f'{self}: Counting types: {type_counts}')
        devlogger.debug(f'{self}: Counting rb types: {rbtc}')
        
    def _updateParticleOrder(self):
        ## Create ordered list
        self.particles = [p for p in self if not p.rigid]
        self.rigid_bodies = list(sorted([p for p in self if p.rigid], key=lambda x: x.type_))
        # self.particles = sorted(particles, key=lambda p: (p.type_, p.idx))
        
        ## Update particle indices
        idx = 0
        for p in self.particles:
            p.idx = idx
            idx = idx+1

        ## Add attached particle indices
        # attach particles
        for j,rb in enumerate(self.rigid_bodies):
            for p in rb.attached_particles:
                p.idx = idx
                idx = idx+1
        
        ## TODO recurse through childrens' group_sites
        for g in self.group_sites:
            g.idx = idx
            idx = idx + 1
            
        # self.initialCoords = np.array([p.initialPosition for p in self.particles])

    def useNonbondedScheme(self, nbScheme, typeA=None, typeB=None):
        """ deprecated """
        logger.warning('useNonbondedScheme is deprecated! Please update your code to use `add_nonbonded_interaction`')
        self.add_nonbonded_interaction(nbScheme, typeA, typeB)

    def add_nonbonded_interaction(self, nonbonded_potential, typeA=None, typeB=None):
        self.nonbonded_interactions.append( [nonbonded_potential, typeA, typeB] )
        if typeA != typeB:
            self.nonbonded_interactions.append( [nonbonded_potential, typeB, typeA] )

    def prepare_for_simulation(self):
        ...
    
    def getParticleTypesAndCounts(self):
        """ Includes rigid body-attached particles """
        ## TODO: remove(?)
        if self.type_counts is None:
            self._countParticleTypes()
            self._updateParticleOrder()

        return sorted( self.type_counts.items(), key=lambda x: x[0] )

    def _particleTypePairIter(self):
        typesAndCounts = self.getParticleTypesAndCounts()
        i_skipped = 0
        for i in range(len(typesAndCounts)):
            t1,(n1,rb1) = typesAndCounts[i]
            if n1+rb1 == 0:
                i_skipped += 1
                continue
            j_skipped = 0
            for j in range(i,len(typesAndCounts)):
                t2,(n2,rb2) = typesAndCounts[j]
                if n2+rb2 == 0:
                    j_skipped += 1
                    continue
                if n2 == 0: continue
                yield( [i-i_skipped,j-i_skipped-j_skipped,t1,t2] )

    def dimensions_from_structure( self, padding_factor=1.5, isotropic=False ):
        raise(NotImplementedError)

    def simulate(self, output_name, **kwargs):
        ## split kwargs
        sim_kws = ['output_directory', 'directory', 'log_file', 'binary', 'num_procs', 'dry_run', 'configuration', 'replicas']
        sim_kwargs = {kw:kwargs[kw] for kw in sim_kws if kw in kwargs}
        engine_kwargs = {k:v for k,v in kwargs.items() if k not in sim_kws}
        engine = ArbdEngine(**engine_kwargs)
        return engine.simulate(self, output_name, **sim_kwargs)
