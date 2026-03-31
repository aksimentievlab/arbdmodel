import numpy as np
from pathlib import Path
from copy import copy, deepcopy
from glob import glob
import MDAnalysis as mda

from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm

from .logger import devlogger, logger, get_resource_path
from .interactions import NullPotential
from . import Transformable, Parent, Group
from .config import SimConf
from .core_objects import GroupSite
from .engine import ArbdEngine, NamdEngine
from .coords import calculate_dimensions_from_cell_vectors


class PdbModel(Transformable, Parent):
    """
    A molecular model class that represents a system of particles and rigid bodies.
    
    This class extends both Transformable and Parent to provide a complete molecular
    modeling framework. It manages particles, rigid bodies, and their interactions,
    and can write PDB format files for visualization and analysis.
    
    Parameters
    ----------
    children : list, optional
        List of child objects (particles, groups, etc.) to include in the model.
        Default is None.
    dimensions : array-like, optional
        Box dimensions [x, y, z] for the simulation system. Default is None.
    remove_duplicate_bonded_terms : bool, optional
        Whether to remove duplicate bonded interaction terms. Default is False.
        
    Attributes
    ----------
    dimensions : array-like
        The simulation box dimensions.
    particles : list
        List of non-rigid particles in the model.
    rigid_bodies : list
        List of rigid body objects in the model.
    cacheInvalid : bool
        Flag indicating whether cached data needs to be updated.
        
    Methods
    -------
    write_pdb(filename, beta_from_fixed=False)
        Write the model structure to a PDB format file.
        
    Examples
    --------
    >>> model = PdbModel(dimensions=[100, 100, 100])
    >>> model.write_pdb("output.pdb")
    """

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
            formatString = "ATOM {idx:>6.6s} {name:^4.4s} {resname:3.3s} {chain:1.1s}{resid:<5.5s}   {x:8.8s}{y:8.8s}{z:8.8s}{occupancy:6.2f}{beta:6.2f}  {charge:2s}{segname:>6s}\n"

            _used_chains = set()
            last_data = None
            current_chain = None
            def _get_chain_id(c):
                nonlocal _used_chains
                def _next(c):
                    if c >= 'Z': return '0'
                    elif c < 'A' and c >= '9': return 'A'
                    return chr(ord(c)+1)
                def _valid(c):
                    return (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9')

                c = c.upper()
                while (c in _used_chains) or not _valid(c):
                    c = _next(c)
                _used_chains.add(c)
                if len(_used_chains) >= 36:
                    _used_chains = {c}
                return c

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
                for d in dig:
                    if d > 7: raise NotImplementedError('arbdmodel is unable to write a PDB with such large dimensions; consider whether shifting your object or scaling dimensions can be used as a workaround')
                fs = ["{: >%d.%df}" % (8,7-d) for d in dig]
                x,y,z = [f.format(x) for f,x in zip(fs,pos)] 
                data['x'] = x
                data['y'] = y
                data['z'] = z
                if data['resid'] > 9999:
                    data['resid'] = ((data['resid']-1) % 9999)+1
                # data['charge'] = str(int(data['charge']))
                data['charge'] = '' # filling this field causes issues with some versions of MDAnalysis reader
                data['resid'] = "{: >4d}".format(data['resid'])

                if last_data is None or data['resid'] < last_data['resid'] or data['chain'] != last_data['chain']:
                    logger.info(last_data)
                    logger.info(data)
                    _chain = _get_chain_id(data['chain'])
                _old_chain = data['chain']
                data['chain'] = _chain
                fh.write( formatString.format(**data) )
                data['chain'] = _old_chain
                last_data = data

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
                 remove_duplicate_bonded_terms=True, buffer_factor=1.2, skip_type_recount=False,
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
            cell_vectors = [
                [dimensions[0], 0, 0],
                [0, dimensions[1], 0],
                [0, 0, dimensions[2]]
            ]
            
        # Set default cell origin if not provided
        if cell_origin is None:
            if dimensions is not None:
                cell_origin = [-0.5 * d for d in dimensions]
            else:
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
        self.skip_type_recount = skip_type_recount

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
        ## Code to create new type name to circumvent issues due to matching types that are not equal
        _equivalent_types = dict()
        for t1 in self.type_counts.keys():
            for t2 in other_model.type_counts.keys():
                if t1.name == t2.name and t1.__eq__(t2, check_equal=False) and t2 is not t1:
                    _equivalent_types[t2.name] = t1

        for t2 in other_model.type_counts.keys():
            if t2.parent is not None:
                key = t2.parent.name
                if key in _equivalent_types:
                    t2.parent = _equivalent_types[key]
                    
        ## Combine interactions
        for i, tA, tB in other_model.nonbonded_interactions:
            if tA is not None and tA.name in _equivalent_types:
                tA = _equivalent_types[tA.name]
            if tB is not None and tB.name in _equivalent_types:
                tB = _equivalent_types[tB.name]
            devlogger.debug(f'Combining model interactions {i} {tA} {tB}')
            self.add_nonbonded_interaction(i,tA,tB)

        if len(_equivalent_types) > 0:
            for p in other_model:
                key = p.type_.name
                if key in _equivalent_types:
                    p.type_ = _equivalent_types[key]

        # for g in other_model.children:
        #     self.update(g, copy=copy)
        for attr in 'children bonds angles dihedrals bondXY impropers exclusions vector_angles bond_angles product_potentials group_sites'.split():
            # self.__setattr__(attr, self.__getattribute__(attr) + other_model.__getattribute__(attr))
            self.__getattribute__(attr).extend(other_model.__getattribute__(attr))

        for obj in other_model.children:
            if obj.parent is other_model:
                obj.parent = self

        self._clear_types()
        
        ## Combine configurations
        self.configuration = other_model.configuration.combine(self.configuration, policy='best', warn=True)

    def assign_IBI_degrees_of_freedom(self):

        """ Convenience routine that adds degrees of freedom to
        corresponding IBI potentials """

        from .ibi import BondDof, AngleDof, DihedralDof, PairDistributionDof, ThreeDimensionalDensityDoF

        self.bonded_ibi_potentials = set()
        self.nonbonded_ibi_potentials = set()
        self.grid_ibi_potentials = set()

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
        # type_to_particles[None] = [p for p in self]

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
            if tA is None or tB is None:
                raise NotImplementedError
            dof = PairDistributionDof( type_to_particles[tA], type_to_particles[tB], exclusions=ex,
                                       range_=pot.range_, resolution=pot.resolution)
            pot.degrees_of_freedom.append( dof )

            if pot not in self.nonbonded_ibi_potentials:
                self.nonbonded_ibi_potentials.add( pot )

        for t in set([p.type_ for p in self]):
            grid_potentials = getattr(t, 'grid_potentials', [])
            for entry in grid_potentials:
                pot = entry[0]
                try:    pot.degrees_of_freedom
                except: continue
                dof = ThreeDimensionalDensityDoF( *type_to_particles[t] )
                pot.degrees_of_freedom.append( dof )
                if pot not in self.grid_ibi_potentials:
                    self.grid_ibi_potentials.add( pot )

    def load_target_IBI_distributions(self):
        raise NotImplementedError

    def _get_IBI_universe(self, iteration, directory='.', replicas=1):
        name = f'ibi-{iteration:03d}'
        psf = '{}/{}.psf'.format(directory,name)
        if replicas == 1:
            dcds = [f'{directory}/output/{name}.dcd']
        else:
            dcds = [f'{directory}/output/{name}.{i}.dcd' for i in range(replicas)]
        dcds = [d for d in dcds if Path(d).exists()]
        if len(dcds) == 0: raise ValueError(f'Expected to find dcds with prefix {directory}/output/{name}')
        cg_u = mda.Universe(psf,*dcds)
        return cg_u

    @property
    def IBI_potentials(self):
        return list(self.bonded_ibi_potentials)+list(self.nonbonded_ibi_potentials)+list(self.grid_ibi_potentials)

    def run_IBI(self, iterations, directory = '.', engine = None, replicas = 1, run_minimization = True, first_iteration=1, target_universe = None, target_trajectory = None):
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
        This method only acts on potentials included in self.IBI_potentials and
        will call self.assign_IBI_degrees_of_freedom() to include all IBI
        potentials in the list if it is empty. Manual management is required to
        restrict IBI optimization to only select potentials. For example, bonded
        potentials are often optimized prior to nonbonded interactions.

        """
        try:
            assert( len(self.IBI_potentials) > 0 )
        except:
            logger.info('Model does not appear to contain IBI potentials; calling assign_IBI_degrees_of_freedom()')
            self.assign_IBI_degrees_of_freedom()
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

        _potentials = self.IBI_potentials

        ## Set potential root_directory to be inside directory if it's relative
        if directory != '.': raise NotImplementedError('Running IBI in another directory not yet supported')
        # note: this can most readily be solved with contextmanager in util.py

        for p in _potentials:
            p.root_directory = directory

        if target_universe is not None:
            with logging_redirect_tqdm(loggers=[logger,devlogger]):
                for potential in tqdm(_potentials, desc='Calculating target distributions'):
                    # import cProfile, pstats, io
                    # from pstats import SortKey
                    # pr = cProfile.Profile()
                    # pr.enable()

                    potential.get_target_distribution(target_universe, trajectory=target_trajectory)

                    # pr.disable()
                    # s = io.StringIO()
                    # sortby = SortKey.CUMULATIVE
                    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                    # ps.print_stats()
                    # print(s.getvalue())
                    # import ipdb; ipdb.set_trace()

        cg_u = None
        if first_iteration > 1:
            cg_u = self._get_IBI_universe(first_iteration-1, replicas=replicas)
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
                    # import cProfile, pstats, io
                    # from pstats import SortKey
                    # pr = cProfile.Profile()
                    # pr.enable()

                    try:    p.write_cg_potential(cg_u, tol=p.tol, box=box)
                    except: p.write_cg_potential(cg_u, box=box)

                    # pr.disable()
                    # s = io.StringIO()
                    # sortby = SortKey.CUMULATIVE
                    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                    # ps.print_stats()
                    # print(s.getvalue())
                    # import ipdb; ipdb.set_trace()

            if i == 1 and run_minimization:
                logger.info(f'Running brief simulation with small timestep')
                ts0 = engine._get_combined_conf(self).timestep
                engine.simulate( self,
                                 output_name = 'ibi-min', directory = directory,
                                 timestep = ts0/100,
                                 num_steps = 10000, output_period=1000 )

                restart_file = f'output/ibi-min.restart'

            name = f'ibi-{i:03d}'
            engine.simulate( self,
                             output_name = name, directory = directory,
                             restart_file = restart_file,
                             replicas = replicas )

            restart_file = f'output/{name}{".0" if replicas > 1 else ""}.restart'
            cg_u = self._get_IBI_universe(i, replicas=replicas)

            with logging_redirect_tqdm(loggers=[logger,devlogger]):
                for p in tqdm(_potentials, desc='Extracting CG distributions'):
                    p.get_cg_distribution(cg_u, box=box, recalculate=False)
                    p.iteration += 1

    def get_IBI_data(self, iteration, universe=None, replicas=1):
        """ Returns list of tuples containing, the potential object, CG distribution bin centers, CG distribution values, tabulated potential coordinates, and tabulated potential values (in kcal/mol) for the provided iteration """
        ret_list = []
        if universe is None: universe = self._get_IBI_universe(iteration, replicas=replicas)
        for p in self.IBI_potentials:
            old_iteration = p.iteration
            try:
                p.iteration=iteration
                bins,dist =  p.get_cg_distribution(universe)
                bin_centers = bins[:-1] + 0.5*(bins[1]-bins[0])
                rmin,rmax = p.range_
                r = np.arange(rmin, rmax+p.resolution, p.resolution)
                ret_list.append( (p, bin_centers, dist, r, p.potential(r)) )
            finally:
                p.iteration = old_iteration
        return ret_list

    def update(self, group , copy=False):
        assert( isinstance(group, Group) )
        if copy:
            group = deepcopy(group)
        group.parent = self
        self.add(group)
        
    def _get_nonbonded_interaction(self, typeA, typeB):
        def __order_types(A,B):
            if A is None: return B,A
            elif B is None: return A,B
            elif A < B: return A,B
            else: return B,A
        typeA,typeB = __order_types(typeA,typeB)

        def __get_generations(t):
            if t.parent: return 1+__get_generations(t.parent)
            return 0
        num_gens = max([__get_generations(t) for t in (typeA,typeB)])
        for consider_wild in (False,True):
            for gens in range(num_gens+1):
                for s,A,B in self.nonbonded_interactions:
                    A,B = __order_types(A,B)
                    if A is None or B is None:
                        if not consider_wild: continue
                        if A is None and B is None:
                            return s
                        elif A is None:
                            if typeA.is_same_type(B, consider_parent_generations=gens) or typeB.is_same_type(B, consider_parent_generations=gens):
                                return s
                        elif B is None:
                            if typeA.is_same_type(A, consider_parent_generations=gens) or typeB.is_same_type(A, consider_parent_generations=gens):
                                return s
                    elif typeA.is_same_type(A, consider_parent_generations=gens) and typeB.is_same_type(B, consider_parent_generations=gens):
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
        if not self.skip_type_recount:
            self.type_counts = None # force recount
        self.getParticleTypesAndCounts()
    
    def getParticleTypesAndCounts(self):
        """ Includes rigid body-attached particles """
        ## TODO: remove(?)
        if self.type_counts is None:
            self._countParticleTypes()
            self._updateParticleOrder()

        return sorted( self.type_counts.items(), key=lambda x: x[0] )
    
    def assign_index_to_types(self):
        i_skipped = 0
        for i,(t,(n,rb)) in enumerate(self.getParticleTypesAndCounts()):
            if n+rb == 0:
                i_skipped += 1
                continue
            t.type_idx = i-i_skipped

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
                yield( [i-i_skipped,j-i_skipped-j_skipped,t1,t2] )

    def dimensions_from_structure( self, padding_factor=1.5, isotropic=False ):
        raise(NotImplementedError)

    def simulate(self, output_name, **kwargs):
        """
        Simulate the model using the specified engine.
        
        This method prepares the simulation configuration and delegates the actual
        simulation to the engine. It supports various simulation parameters through
        keyword arguments.

        Parameters
        ----------
        output_name : str
            The base name for the simulation output files.
        **kwargs : dict, optional
            Additional keyword arguments for the simulation configuration.
        """
        
        ## split kwargs
        sim_kws = ['output_directory', 'directory', 'log_file', 'binary', 'num_procs', 'dry_run', 'configuration', 'replicas']
        sim_kwargs = {kw:kwargs[kw] for kw in sim_kws if kw in kwargs}
        engine_kwargs = {k:v for k,v in kwargs.items() if k not in sim_kws}
        engine = ArbdEngine(**engine_kwargs)
        return engine.simulate(self, output_name, **sim_kwargs)
