## Import packages
from pathlib import Path
import numpy as np
import os, sys, subprocess
import platform
from abc import abstractmethod, ABCMeta
import shutil
from .core_objects import GroupSite
from .logger import devlogger, logger, get_resource_path
from .interactions import NullPotential
from .config import SimConf


""" SimEngines
Abstract classes for running simulations with different engines.
SimEngine is the base class for all simulation engines.
"""
class SimEngine(metaclass=ABCMeta):
    """ Abstract class for running a simulation of a model """
    def __init__(self, configuration=None):
        self.configuration = configuration

    @property
    @abstractmethod
    def default_binary(self):
        ...

    @abstractmethod
    def _generate_command_string(self, binary, output_name, output_directory, gpu=0, replicas=1):
        ...

    @abstractmethod
    def write_simulation_files(self, model, output_name):
        ...

    @abstractmethod
    def get_default_conf(self):
        ...
        
    def simulate(self, model, output_name, output_directory='output',
                 directory='.', log_file=None,
                 binary=None, num_procs=None, dry_run = False, configuration = None, replicas = 1, **conf_params):

        ## TODO: Allow _get_combined_conf to take certain parameters as arguments, or otherwise refactor to make this more elegant
        gpu = self._get_combined_conf(model, **conf_params).gpu
        assert(type(gpu) is int)

        if num_procs is None:
            import multiprocessing
            num_procs = max(1,multiprocessing.cpu_count()-1)

        d_orig = os.getcwd()
        try:
            model._d_orig = d_orig
            if not os.path.exists(directory):
                os.makedirs(directory)
            os.chdir(directory)

            model.prepare_for_simulation()

            if output_directory == '': output_directory='.'

            if dry_run:
                if binary is None: binary=self.default_binary
            else:
                binary = self._get_binary(binary)

            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            elif not os.path.isdir(output_directory):
                raise Exception("output_directory '%s' is not a directory!" % output_directory)

            self.write_simulation_files(model, output_name, configuration, **conf_params)

            ## http://stackoverflow.com/questions/18421757/live-output-from-subprocess-command
            cmd = self._generate_command_string(binary, output_name, output_directory, num_procs, gpu, replicas)

            if dry_run:
                logger.info(f'Run with: {" ".join(cmd)}')
            else:
                logger.info(f'Running {self.default_binary} with: {" ".join(cmd)}')
                if log_file is None or (hasattr(log_file,'write') and callable(log_file.write)):
                    fd = sys.stdout if log_file is None else log_file
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=False) # , universal_newlines=True)
                    ## re-open to get \r recognized as new line
                    with open(os.dup(process.stdout.fileno()), newline='') as nice_stdout:
                        for line in nice_stdout:
                            try:
                                fd.write(line)
                            except:
                                print("WARNING: could not encode line; your locale might not be set correctly")
                            fd.flush()
                else:
                    with open(log_file,'w') as fd:
                        process = subprocess.Popen(cmd, stdout=log_file, universal_newlines=True)
                        process.communicate()

        except:
            raise
        finally:
            del model._d_orig
            os.chdir(d_orig)

    def _get_binary(self, binary=None):
        if binary is None:
            from .binary_manager import BinaryManager
            binary_name = self.default_binary
            binary = BinaryManager.get_binary_path(binary_name)

        if binary is None:
            for path in os.environ["PATH"].split(os.pathsep):
                path = path.strip('"')
                fname = os.path.join(path, self.default_binary)
                if os.path.isfile(fname) and os.access(fname, os.X_OK):
                    binary = fname
                    break
        if binary is None: raise Exception("{} was not found".format(self.default_binary))

        if not os.path.exists(binary):
            raise Exception("{} was not found".format(self.default_binary))
        if not os.path.isfile(binary):
            raise Exception("{} was not found".format(self.default_binary))
        if not os.access(binary, os.X_OK):
            raise Exception("{} is not executable".format(self.default_binary))

        return binary

    def _get_combined_conf(self, model, **conf_params):
        conf = self.get_default_conf() 
        conf = conf.combine(model.configuration)
        conf = conf.combine(self.configuration)
        conf = conf.combine(SimConf(**conf_params))
        return conf

    """ TODO: Remove
    # def _configuration_as_dict(self, model):
    #     params = dict()
    #     for o in (self.configuration, self, model):
    #         for k in _get_properties_and_dict_keys(o):
    #             params[k] = o.__get_attribute__(k)
    #     return params
    """

class ArbdEngine(SimEngine):
    """
    Interface to ARBD simulation engine.
    The ArbdEngine class provides an interface to the ARBD (Atomically-Resolved Brownian Dynamics) 
    simulation engine. It handles the generation of simulation configuration files, particle coordinates,
    and various potential files required for ARBD simulations.

    Parameters
    ----------
    extra_bd_file_lines (str): Additional lines to be added to the BD configuration file.
    configuration (SimConf): The simulation configuration object.
    num_particles (int): Number of particles in the system.
    particles (list): List of particles in the system.
    type_counts (dict): Count of each particle type.
    cacheUpToDate (bool): Flag indicating if the cache is current.
    potential_directory (str): Directory for storing potential files.
    rb_type_dirs (dict): Directories for rigid body types.

    """
    def __init__(self, extra_bd_file_lines="", configuration=None, **conf_params):
        self.extra_bd_file_lines = extra_bd_file_lines
        
        if configuration is None: 
            configuration = SimConf(**conf_params)
        SimEngine.__init__(self, configuration)

        self.num_particles = 0
        self.particles = []     # TODO decide if this should belong here, or with model
        self.type_counts = None # TODO decide if this should belong here, or with model

        self._written_bond_files = dict()        

        self.cacheUpToDate = False

    @property
    def default_binary(self):
        return 'arbd'

    def _generate_command_string(self, binary, output_name, output_directory, num_procs=None, gpu=0, replicas=1):
        cmd = [binary, '-g', "%d" % gpu]
        if replicas > 1:
            cmd = cmd + ['-r',replicas]
        cmd = cmd + ["%s.bd" % output_name, "%s/%s" % (output_directory, output_name)]
        cmd = tuple(str(x) for x in cmd)
        return cmd

    def get_default_conf(self):
        conf = SimConf(num_steps=1e5, output_period=1e3,
                 integrator='MD', timestep=20e-6, thermostat='Langevin', barostat=None,
                 temperature=295, pressure=1,
                 cutoff=50, pairlist_distance=None, decomp_period=40, gpu = 0,
                 seed=None, restart_file=None)
        return conf
    
    # -------------------------- #
    # Methods for printing model #
    # -------------------------- #

    def write_simulation_files(self, model, output_name, configuration=None, **conf_params):
        """
        Write simulation files for a model.
        This method generates all necessary files for simulation, including system 
        topology files (PSF, PDB), particle coordinates, and various potential files.

        Parameters
        ----------
        model : object
            The model object containing all necessary information about the system.
        output_name : str
            Base name for output files.
        configuration : dict, optional
            Configuration dictionary for the model. If None, a configuration will be 
            generated using _get_combined_conf.
        **conf_params : dict
            Additional configuration parameters to pass to _get_combined_conf if
            configuration is None.

        Note
        -----
        The method creates directory structure for potentials and rigid body types
        if they don't exist. It generates the following files:
        - PSF and PDB topology files
        - Particle configuration file
        - Various potential files (restraint, bond, angle, dihedral, etc.)
        - Rigid body coordinate and configuration files
        - Creates directories if they don't exist
        - Sets self.potential_directory and self.rb_type_dirs attributes
        - Writes multiple files to disk
        """
        if configuration is None:
            configuration = self._get_combined_conf(model, **conf_params)
        
        ## TODO: save and reference directories and prefixes using member data
        d = self.potential_directory = "potentials"
        if not os.path.exists(d):
            os.makedirs(d)
        rb_type_dirs = {}
        for rbk, num in model.rigid_body_type_counts:
            if num > 0:
                rbt = model.rigid_body_index[rbk]
                rb_dir = rbt.name  # Create top-level directory named after the RigidBodyType
                if not os.path.exists(rb_dir):
                    os.makedirs(rb_dir)
                rb_type_dirs[rbt.name] = rb_dir
        
        main_potentials_dir=self.potential_directory 
        self.rb_type_dirs = rb_type_dirs

        model.write_psf( output_name+'.psf' )
        model.write_pdb( output_name+'.pdb' )

        self._write_particle_file(model, output_name + ".particles.txt", configuration)
        
        self._write_restraint_file(model, f"{main_potentials_dir}/{output_name}.restraint.txt")
        self._write_bond_file(model, f"{main_potentials_dir}/{output_name}.bond.txt")
        self._write_angle_file(model, f"{main_potentials_dir}/{output_name}.angle.txt")
        self._write_dihedral_file(model, f"{main_potentials_dir}/{output_name}.dihedral.txt")
        self._write_vector_angle_file(model, f"{main_potentials_dir}/{output_name}.vecangle.txt")
        self._write_exclusion_file(model, f"{main_potentials_dir}/{output_name}.exclusion.txt")
        self._write_bond_angle_file(model, f"{main_potentials_dir}/{output_name}.bond-angle.txt")
        self._write_product_potential_file(model, f"{main_potentials_dir}/{output_name}.product_potential.txt")
        self._write_group_sites_file(model, f"{main_potentials_dir}/{output_name}.group_sites.txt")
        self._write_rb_coordinate_file(model, configuration, f'{output_name}.rbcoords.txt')
        
        self._write_potential_files(model, output_name, directory=main_potentials_dir, configuration=configuration)
        
        self._write_rb_attached_particles_files(model, output_name, configuration)
        
        self._write_conf(model, output_name, configuration)
        ## , numSteps=numSteps, outputPeriod=outputPeriod, restart_file=restart_file )

    def _write_particle_file(self, model, filename, configuration=None, **conf_params):
        if configuration is None:
            configuration = self._get_combined_conf(model, **conf_params)

        with open(filename,'w') as fh:
            if configuration.integrator in ('Brown', 'Brownian', 'BD'):
                for p in model.particles:
                    data = tuple([p.idx,p.type_.name] + [x for x in p.get_collapsed_position()])
                    fh.write("ATOM %d %s %f %f %f\n" % data)
            else:
                for p in model.particles:
                    data = [p.idx,p.type_.name] + [x for x in p.get_collapsed_position()]
                    try:
                        data = data + p.momentum
                    except:
                        try:
                            data = data + p.velocity*p.mass
                        except:
                            data = data + [0,0,0]
                    fh.write("ATOM %d %s %f %f %f %f %f %f\n" % tuple(data))

    def _write_rb_attached_particles_files(self, model, output_name, configuration=None, **conf_params):
        if configuration is None:
            configuration = self._get_combined_conf(model, **conf_params)
        if len(model.rigid_bodies) > 0:
            for rbk, num in model.rigid_body_type_counts:
                rbt = model.rigid_body_index[rbk]
                devlogger.debug(f'Writing attached particles file for rigid body type {rbt}')
                if num > 0 and len(rbt.attached_particles) > 0:
                    # Use RB-specific directory
                    rb_dir = self.rb_type_dirs.get(rbt.name)
                    if not rb_dir:
                        logger.warning(f"No directory found for RigidBodyType {rbt.name}, using default")
                        rb_dir = self.potential_directory
                    
                    # Create the attached particles file inside the RB directory
                    f = os.path.join(rb_dir, f"attached_particles.txt")
                    rbt._attached_particles_filename = f
                    with open(f, 'w') as fh:
                        for p in rbt.attached_particles:
                            x, y, z = p.position
                            fh.write(f'{p.type_.name} {x} {y} {z}\n')

    def _write_rigid_group_file(self, model, filename, groups):
        raise Exception('Deprecated')
        with open(filename,'w') as fh:
            for g in groups:
                fh.write("#Group\n")
                try:
                    if len(g.trans_damping) != 3: raise
                    fh.write(" ".join(str(v) for v in g.trans_damping) + " ")
                except:
                    raise ValueError("Group {} lacks 3-value 'trans_damping' attribute")
                try:
                    if len(g.rot_damping) != 3: raise
                    fh.write(" ".join(str(v) for v in g.rot_damping) + " ")
                except:
                    raise ValueError("Group {} lacks 3-value 'rot_damping' attribute")
                fh.write("{}\n".format(len(g)))
                particles = [p for p in g]

                def chunks(l, n):
                    """Yield successive n-sized chunks from l."""
                    for i in range(0, len(l), n):
                        yield l[i:i + n]

                for c in chunks(particles,8):
                    fh.write(" ".join(str(p.idx) for p in c) + "\n")

    def _write_potential_files(self, model, prefix, directory = "potentials", configuration=None, **conf_params):
        try: 
            os.makedirs(directory)
        except OSError:
            if not os.path.isdir(directory):
                raise

        if configuration is None:
            configuration = self._get_combined_conf(model, **conf_params)

        path_prefix = "{}/{}".format(directory,prefix)
        self._write_nonbonded_parameter_files( model, path_prefix + "-nb", configuration )
                
    def _write_nonbonded_parameter_files(self, model, prefix, configuration=None, **conf_params):
        if configuration is None:
            configuration = self._get_combined_conf(model, **conf_params)

        model._nonbonded_interaction_files = [] # clear old nb files

        x = np.arange(0, configuration.cutoff)
        for i,j,t1,t2 in model._particleTypePairIter():
            interaction = model._get_nonbonded_interaction(t1,t2)
            if interaction is None: interaction = NullPotential()
            try:
                f = interaction.filename(types=(t1,t2))
            except:
                f = "%s.%s-%s.dat" % (prefix, t1.name, t2.name)
                logger.debug(f'_write_nonbonded_parameter_files could not find filename for {interaction}; using default {f}')

            devlogger.debug(f'_write_nonbonded_parameter_files: {i}, {j}, {t1}, {t2}, {interaction}')
            old_range = interaction.range_
            if not isinstance(interaction,NullPotential):
                interaction.range_ = [0, configuration.cutoff]
            interaction.write_file(f, (t1, t2))
            interaction.range_ = old_range 

            model._nonbonded_interaction_files.append(f)
        devlogger.debug(f'model._nonbonded_interaction_files: {model._nonbonded_interaction_files}')

    def _write_restraint_file( self, model, filename ):
        self._restraint_filename = filename
        with open(self._restraint_filename,'w') as fh:
            for i,restraint in model.get_restraints():
                item = [i.idx]
                if len(restraint) == 1:
                    item.append(restraint[0])
                    if isinstance(i, GroupSite):
                        item.extend(i.get_center())
                    else:
                        item.extend(i.get_collapsed_position())
                elif len(restraint) == 2:
                    item.append(restraint[0])
                    item.extend(restraint[1])
                elif len(restraint) == 5:
                    item.extend(restraint)
                fh.write("RESTRAINT %d %f %f %f %f\n" % tuple(item))

    def _write_bond_file( self, model, filename ):
        self._bond_filename = filename
        for b in list( set( [b for i,j,b,ex in model.get_bonds()] ) ):
            if type(b) is not str and not isinstance(b, Path):
                b.write_file()

        with open(self._bond_filename,'w') as fh:
            for i,j,b,ex in model.get_bonds():
                try:
                    bfile = b.filename()
                except:
                    bfile = str(b)
                item = (i.idx, j.idx, bfile)
                if ex:
                    fh.write("BOND REPLACE %d %d %s\n" % item)
                else:
                    fh.write("BOND ADD %d %d %s\n" % item)

    def _write_angle_file( self, model, filename ):
        self._angle_filename = filename
        for b in list( set( [b for i,j,k,b in model.get_angles()] ) ):
            if type(b) is not str and not isinstance(b, Path):
                b.write_file()

        with open(self._angle_filename,'w') as fh:
            for b in model.get_angles():
                try:
                    bfile = b[-1].filename()
                except:
                    bfile = str(b[-1])
                item = tuple([p.idx for p in b[:-1]] + [bfile])
                fh.write("ANGLE %d %d %d %s\n" % item)

    def _write_dihedral_file( self, model, filename ):
        self._dihedral_filename = filename
        for b in list( set( [b for i,j,k,l,b in model.get_dihedrals()] ) ):
            if type(b) is not str and not isinstance(b, Path):
                b.write_file()

        with open(self._dihedral_filename,'w') as fh:
            for b in model.get_dihedrals():
                try:
                    bfile = b[-1].filename()
                except:
                    bfile = str(b[-1])
                item = tuple([p.idx for p in b[:-1]] + [bfile])
                fh.write("DIHEDRAL %d %d %d %d %s\n" % item)

    def _write_vector_angle_file( self, model, filename ):
        self._vector_angle_filename = filename

        for b in list( set( [b for i,j,k,l,b in model.get_vector_angles()] ) ):
            if type(b) is not str and not isinstance(b, Path):
                b.write_file()

        if len(model.vector_angles) > 0:
            with open(self._vector_angle_filename,'w') as fh:
                for b in model.get_vector_angles():
                    p = b[-1]
                    try:
                        bfile = p.filename()
                    except:
                        bfile = str(p)
                    item = tuple([p.idx for p in b[:-1]] + [bfile])
                    fh.write("VECANGLE %d %d %d %d %s\n" % item)

    def _write_exclusion_file( self, model, filename ):
        self._exclusion_filename = filename
        with open(self._exclusion_filename,'w') as fh:
            for ex in model.get_exclusions():
                item = tuple(int(p.idx) for p in ex)
                fh.write("EXCLUDE %d %d\n" % item)

    def _write_bond_angle_file( self, model, filename ):
        self._bond_angle_filename = filename
        if len(model.bond_angles) > 0:
            with open(self._bond_angle_filename,'w') as fh:
                for b in model.get_bond_angles():
                    bfiles = []
                    for p in b[-1]:
                        try:
                            bfile = p.filename()
                        except:
                            bfile = str(p)
                        bfiles.append(bfile)
                    item = tuple([p.idx for p in b[:-1]] + bfiles)
                    fh.write("BONDANGLE %d %d %d %d %s %s %s\n" % item)

    def _write_product_potential_file( self, model, filename ):
        self._product_potential_filename = filename
        if len(model.product_potentials) > 0:
            with open(self._product_potential_filename,'w') as fh:
                for pot in model.get_product_potentials():
                    line = "PRODUCTPOTENTIAL "
                    for ijk_tb in pot:
                        ijk = ijk_tb[:-1]
                        tb = ijk_tb[-1]
                        if type(tb) is tuple or type(tb) is list:
                            if len(tb) != 2: raise ValueError("Invalid product potential")
                            type_,b = tb
                            if type(type_) is not str: raise ValueError("Invalid product potential: unrecognized specification of potential type")
                        else:
                            type_ = ""
                            b = tb
                        if type(b) is not str and not isinstance(b, Path):
                            b.write_file()
                        try:
                            bfile = b.filename()
                        except:
                            bfile = str(b)
                        line = line+" ".join([str(x.idx) for x in ijk])+" "
                        line = line+" ".join([str(x) for x in [type_,bfile] if x != ""])+" "
                    fh.write(line)

    def _write_group_sites_file( self, model, filename ):
        self._group_sites_filename = filename
        if len(model.group_sites) > 0:
            with open(self._group_sites_filename,'w') as fh:
                for i,g in enumerate(model.group_sites):
                    assert( i+len(model.particles) == g.idx )
                    ids = " ".join([str(int(p.idx)) for p in g.particles])
                    fh.write("GROUP %s\n" % ids)

    def _write_rb_coordinate_file( self, model, configuration, filename ):
        self._rb_coordinate_filename = filename
        if len(model.rigid_bodies) > 0:
            rb_integrator = configuration.rigid_body_integrator
            if rb_integrator is None: rb_integrator = configuration.integrator
            with open(self._rb_coordinate_filename,'w') as fh:
                for rb in model.rigid_bodies:
                    o = list(rb.applyOrientation(rb.orientation).flatten())
                    if len(o) != 9: raise ValueError('Rigid body orientation should be a 3x3 matrix')
                    fh.write(' '.join(map(str,list(rb.get_collapsed_position()) + o)))
                    if rb_integrator in ('MD','Langevin'):
                        try: fh.write( ' ' + ' '.join([str(x) for x in rb.momentum]) )
                        except:  fh.write( ' 0'*3 )
                        try: fh.write( ' ' + ' '.join([str(x) for x in rb.rotational_momentum]) )
                        except:  fh.write( ' 0'*3 )
                    fh.write('\n')


    def _write_conf(self, model, prefix, configuration):
        # num_steps=100000000, output_period=10000, restart_file=None,
        ## TODO: raise exception if _write_potential_files has not yet been called
        filename = f'{prefix}.bd'

        ## Create helper function
        def _fix_path(filename, rb_type=None):
            if rb_type and rb_type in self.rb_type_dirs:
                # First, check if this is an existing file within the RB directory
                rb_dir = self.rb_type_dirs[rb_type]
                basename = os.path.basename(str(filename))
                rb_path = os.path.join(rb_dir, basename)
                if os.path.exists(rb_path):
                    return rb_path
                
                # Check if this might be a resource we want to place in the RB directory
                if basename.startswith(rb_type) or rb_type in str(filename):
                    return rb_path
            
            abspath = str(filename) if str(filename)[0] == '/' else Path(model._d_orig) / filename
            ret = None
            try: 
                ret = os.path.relpath(str(abspath))
            except:
                devlogger.info(f'Relative path for {filename} not found... using {abspath}')
                ret = abspath
            return str(ret)
        
        ## Build dictionary of parameters
        params = dict()
        for k,v in configuration.items():
            params[k] = v

        if configuration.seed is None:
            params['seed']     = f'seed {int(np.random.default_rng().integers(1,99999,1))}'
        else:
            params['seed'] = "seed {:d}".format(configuration.seed)
        params['num_steps']       = int(configuration.num_steps)

        # params['coordinateFile'] = "%s.coord.txt" % prefix
        params['particle_file'] = "%s.particles.txt" % prefix
        if configuration.restart_file is None:
            params['restart_coordinates'] = ""
        else:
            params['restart_coordinates'] = "restartCoordinates %s" % configuration.restart_file

        for k,v in zip('XYZ', model.dimensions):
            params['dim'+k] = v

        if model.origin is None:
            for k,v in zip('XYZ', model.dimensions):
                params['origin'+k] = -v*0.5
        else:
            for k,v in zip('XYZ', model.origin):
                params['origin'+k] = v
             
        if params['pairlist_distance'] is None:
            params['pairlist_distance'] = 10
        else:
            params['pairlist_distance'] -= params['cutoff'] 

        if params['integrator'] == 'MD':
            params['integrator'] = 'Langevin'

        if len(model.rigid_bodies) > 0:
            rb_integrator = params['rigid_body_integrator'] 
            if rb_integrator is None: rb_integrator = params["integrator"]
            params['rigid_body_integrator'] = f'\nRigidBodyDynamicType {rb_integrator}'
            _rbggp = params["rigid_body_grid_grid_period"]
            if _rbggp is not None and _rbggp > 1:
                params['rigid_body_integrator'] += f'\nrigidBodyGridGridPeriod {_rbggp}'
        else:
            params['rigid_body_integrator'] = ''
            
        ## Actually write the file
        with open(filename,'w') as fh:
            fh.write("""{seed}
timestep {timestep}
steps {num_steps}
numberFluct 0                   # deprecated

interparticleForce 1            # other values deprecated
fullLongRange 0                 # deprecated
temperature {temperature}
ParticleDynamicType {integrator}{rigid_body_integrator}

outputPeriod {output_period}
## Energy doesn't actually get printed!
outputEnergyPeriod {output_period}
outputFormat dcd

## Infrequent domain decomposition because this kernel is still very slow
decompPeriod {decomp_period}
cutoff {cutoff}
pairlistDistance {pairlist_distance}

origin {originX} {originY} {originZ}
systemSize {dimX} {dimY} {dimZ}

{extra_bd_file_lines}
\n""".format(extra_bd_file_lines=self.extra_bd_file_lines, **params))
            
            ## Write entries for each type of particle
            for pt,(num,num_rigid) in model.getParticleTypesAndCounts():
                if num+num_rigid == 0: continue
                devlogger.debug(f'Writing configuration for particle type {pt}')
                ## TODO create new particle types if existing has grid
                particleParams = pt.__dict__.copy()
                particleParams['num'] = num
                if configuration.integrator in ('Brown', 'Brownian', 'BD'):
                    try:
                        D = pt.diffusivity
                    except:
                        """ units "k K/(amu/ns)" "AA**2/ns" """
                        D = 831447.2 * configuration.temperature / (pt.mass * pt.damping_coefficient)
                    particleParams['dynamics'] = 'diffusion {D}'.format(D = D)
                elif configuration.integrator in ('MD','Langevin','FusDynamic'):
                    try:
                        gamma = pt.damping_coefficient
                        if gamma is None: raise
                    except:
                        """ units "k K/(AA**2/ns)" "amu/ns" """
                        gamma = 831447.2 * configuration.temperature / (pt.mass*pt.diffusivity)
                    particleParams['dynamics'] = """mass {mass}
transDamping {g} {g} {g}""".format(mass=pt.mass, g=gamma)
                else:
                    raise ValueError("Unrecognized particle integrator '{}'".format(configuration.particle_integrator))
                fh.write("""
particle {name}
num {num}
{dynamics}
""".format(**particleParams))
                if 'grid_potentials' in particleParams:
                    grids = []
                    scales = []
                    boundary_conditions = []
                    for vals in pt.grid_potentials:
                        try: g,s,bc = vals
                        except:
                            logger.warning(f'Failed to unpack {pt}.grid_potentials, presumably due to lack of specified boundary condition... using "dirichlet"')
                            g,s = vals
                            bc = 'dirichlet'
                        grids.append( _fix_path(g) )
                        scales.append(str(s))
                        boundary_conditions.append(bc)

                    fh.write("gridFile {}\n".format(" ".join(grids)))
                    fh.write("gridFileScale {}\n".format(" ".join(scales)))
                    if any([bc != 'dirichlet' for bc in boundary_conditions]):
                        fh.write(f'gridFileBoundaryConditions {" ".join(boundary_conditions)}'+'\n')
                else:
                    fh.write("gridFile {}/null.dx\n".format(self.potential_directory))

                if 'forceXGrid' in particleParams:
                    fh.write(f"forceXGridFile {_fix_path(pt.forceXGrid)}\n")
                if 'forceYGrid' in particleParams:
                    fh.write(f"forceYGridFile {_fix_path(pt.forceYGrid)}\n")
                if 'forceZGrid' in particleParams:
                    fh.write(f"forceZGridFile {_fix_path(pt.forceZGrid)}\n")

                if any(f'force{x}Grid' in particleParams for x in ('X','Y','Z')) and \
                   'forceGridScale' in particleParams:
                    _scale = None
                    if isinstance( pt.forceGridScale, float ):
                        _scale = 3*[pt.forceGridScale]
                    elif len( pt.forceGridScale ) == 1:
                        _scale = 3*[pt.forceGridScale[0]]
                    elif len( pt.forceGridScale ) == 3:
                        _scale = pt.forceGridScale
                    else:
                        raise ValueError(f'Unrecognized format for ParticleType.forceGridScale: "{pt.forceGridScale}"')
                    fh.write(f"forceGridScale {' '.join(map(str,_scale))}")
                    
                if 'rigid_body_potentials' in particleParams:
                    grids = []
                    scales = []
                    for item in pt.rigid_body_potentials:
                        try:    keyword,s = item
                        except: keyword,s = (item,1)
                        fh.write(f"rigidBodyPotential {keyword}\n")
                        if s != 1: raise NotImplementedError('Instead scale rigid body potential')

            ## Write coordinates and interactions
            fh.write("""
## Input coordinates
inputParticles {particle_file}
{restart_coordinates}

## Interaction potentials
tabulatedPotential  1
## The i@j@file syntax means particle type i will have NB interactions with particle type j using the potential in file
""".format(**params))
            for pair,f in zip(model._particleTypePairIter(), model._nonbonded_interaction_files):
                if f is not None:
                    i,j,t1,t2 = pair
                    fh.write("tabulatedFile %d@%d@%s\n" % (i,j,f))

            ## Bonded interactions
            restraints = model.get_restraints()
            bonds = model.get_bonds()
            angles = model.get_angles()
            dihedrals = model.get_dihedrals()
            vector_angles = model.get_vector_angles()
            exclusions = model.get_exclusions()
            bond_angles = model.get_bond_angles()
            prod_pots = model.get_product_potentials()
            # group_sites = model.get_group_sites()
            group_sites = model.group_sites

            if len(bonds) > 0:
                for b in model._get_bond_potentials():
                    try:
                        bfile = b.filename()
                    except:
                        bfile = str(b)
                    fh.write("tabulatedBondFile %s\n" % bfile)

            if len(angles) > 0:
                for b in model._get_angle_potentials():
                    try:
                        bfile = b.filename()
                    except:
                        bfile = str(b)
                    fh.write("tabulatedAngleFile %s\n" % bfile)

            if len(vector_angles) > 0:
                for b in list(set([b for i,j,k,l,b in vector_angles])):
                    try:
                        bfile = b.filename()
                    except:
                        bfile = str(b)
                    fh.write("tabulatedVecangleFile %s\n" % bfile)

            if len(dihedrals) > 0:
                for b in list(set([b for i,j,k,l,b in dihedrals])):
                    try:
                        bfile = b.filename()
                    except:
                        bfile = str(b)
                    fh.write("tabulatedDihedralFile %s\n" % bfile)

            if len(vector_angles) > 0:
                for b in list(set([b for i,j,k,l,b in vector_angles])):
                    try:
                        bfile = b.filename()
                    except:
                        bfile = str(b)
                    fh.write("tabulatedVecangleFile %s\n" % bfile)

            if len(restraints) > 0:
                fh.write("inputRestraints %s\n" % self._restraint_filename)
            if len(bonds) > 0:
                fh.write("inputBonds %s\n" % self._bond_filename)
            if len(angles) > 0:
                fh.write("inputAngles %s\n" % self._angle_filename)
            if len(dihedrals) > 0:
                fh.write("inputDihedrals %s\n" % self._dihedral_filename)
            if len(vector_angles) > 0:
                fh.write("inputVecangles %s\n" % self._vector_angle_filename)
            if len(exclusions) > 0:
                fh.write("inputExcludes %s\n" % self._exclusion_filename)
            if len(vector_angles) > 0:
                fh.write("inputVecangles %s\n" % self._vector_angle_filename)
            if len(bond_angles) > 0:
                fh.write("inputBondAngles %s\n" % self._bond_angle_filename)
            if len(prod_pots) > 0:
                fh.write("inputProductPotentials %s\n" % self._product_potential_filename)
            if len(group_sites) > 0:
                fh.write("inputGroups %s\n" % self._group_sites_filename)

            if len(model.rigid_bodies) > 0:
                for rbi,num in model.rigid_body_type_counts:
                    rbt=model.rigid_body_index[rbi]
                    if num == 0: continue
                    devlogger.debug(f'Writing configuration for rigid body type {rbt}')
                    ## For now, we always convert rigid body diffusivity into mass+damping
                    try:
                        gamma = rbt.damping_coefficient
                    except:
                        """ units "k K/(AA**2/ns)" "dalton/ns" """
                        gamma = 831447.2 * configuration.temperature / (rbt.mass*np.array(rbt.diffusivity))
                    if len(gamma) == 1:
                        logger.warn(f'Using single diffusion coefficient for all motions along all rigid body principal axes for {rbt}')
                        gamma = 3*[gamma]

                    try:
                        gamma_rot = rbt.rotational_damping_coefficient
                    except:
                        """ units "k K/(1/ns)" "AA**2 dalton/ns" """
                        gamma_rot = 831447.2 * configuration.temperature / (np.array(rbt.moment_of_inertia)*np.array(rbt.diffusivity))
                    if len(gamma_rot) == 1:
                        logger.warn(f'Using single rotational diffusion coefficient for all motions along all rigid body principal axes for {pt}')
                        gamma_rot = 3*[gamma_rot]
                    if len(gamma_rot) != 3: raise ValueError('Expected a three-element rotational diffusion coefficient for rigid bodies')
                    fh.write(f"""
rigidBody {rbt.name}
num {num}
mass {rbt.mass}
inertia {' '.join(map(str,rbt.moment_of_inertia))}
transDamping {' '.join(map(str,gamma))}
rotDamping {' '.join(map(str,gamma_rot))}
""")

                    for item in rbt.potential_grids:
                        try:    keyword,g,s = item
                        except: (keyword,g),s = (item,1)

                        g = _fix_path(g,rbt.name)
                        fh.write(f"potentialGrid {keyword} {g}\n")
                        if s != 1: fh.write(f"potentialGridScale {keyword} {s}\n")

                    for item in rbt.charge_grids:
                        try:    keyword,g,s = item
                        except: (keyword,g),s = (item,1)
                        g = _fix_path(g,rbt.name)
                        fh.write(f"densityGrid {keyword} {g}\n")
                        if s != 1: fh.write(f"densityGridScale {keyword} {s}\n")
                    for item in rbt.pmf_grids:
                        try:    keyword,g,s = item
                        except: (keyword,g),s = (item,1)
                        g = _fix_path(g,rbt.name)
                        fh.write(f"gridFile {keyword} {g}\n")
                        if s != 1: fh.write(f"pmfScale {keyword} {s}\n")

                    ## AttachedParticles
                    if len(rbt.attached_particles) > 0:
                        fh.write(f'attachedParticles {rbt._attached_particles_filename}\n')

                fh.write(f'\ninputRBCoordinates {self._rb_coordinate_filename}\n')
                ...
                
        write_null_dx = False
        for pt,(num,num_rb) in model.getParticleTypesAndCounts():
            if num+num_rb == 0: continue
            if "grid_potentials" not in pt.__dict__:
                gridfile = "{}/null.dx".format(self.potential_directory)
                with open(gridfile, 'w') as fh:
                    fh.write("""object 1 class gridpositions counts  2 2 2
origin {originX} {originY} {originZ}
delta  {dimX} 0.000000 0.000000
delta  0.000000 {dimY} 0.000000
delta  0.000000 0.000000 {dimZ}
object 2 class gridconnections counts  2 2 2
object 3 class array type float rank 0 items 8 data follows
0.0	0.0	0.0	
0.0	0.0	0.0	
0.0	0.0	
attribute "dep" string "positions"
object "density" class field 
component "positions" value 1
component "connections" value 2
component "data" value 3
""".format(**params))
                    break

class NamdEngine(SimEngine):
    """ Partial interface to NAMD simulation engine """

    def __init__(self, configuration=None, **conf_params):
        if configuration is None: 
            configuration = SimConf(**conf_params)
        SimEngine.__init__(self, configuration)

        self.num_particles = 0
        self.particles = []     # TODO decide if this should belong here, or with model
        self.type_counts = None # TODO decide if this should belong here, or with model
        self.cacheUpToDate = False

    @property
    def default_binary(self):
        return 'namd2'

    def _generate_command_string(self, binary, output_name, output_directory, num_procs=1, gpu=None, replicas=1):
        cmd = [binary]
        if gpu is not None and len(gpu) > 0:
            cmd.extend(['+devices'] + [','.join([str(int(g)) for g in gpu])])
        cmd.extend([f'+p{str(num_procs)}'])    
        if replicas > 1:
            raise NotImplementedError
        cmd = cmd + [f'{output_name}.namd']
        cmd = tuple(str(x) for x in cmd)
        return cmd

    def get_default_conf(self):
        conf = SimConf(num_steps=1e5, output_period=1e3,
                 integrator='MD', timestep=2e-6, thermostat='Langevin', barostat=None,
                 temperature=295, pressure=1,
                 cutoff=12, pairlist_distance=14, decomp_period=12,
                 seed=None, restart_file=None)
        return conf

    def write_simulation_files(self, model, output_name, configuration=None, write_pqr=False, copy_ff_from=get_resource_path("charmm36.nbfix"), **conf_params):
        if configuration is None:
            configuration = self._get_combined_conf(model, **conf_params)

        model.write_psf( output_name+'.psf' )
        model.write_pdb( output_name+'.pdb' )
        model.write_pdb( output_name + ".fixed.pdb", beta_from_fixed=True )
        if write_pqr: model.write_pqr( output_name + ".pqr" )        

        if copy_ff_from is not None and copy_ff_from != '':
            try:
                shutil.copytree( copy_ff_from, Path(copy_ff_from).stem )
            except FileExistsError:
                pass

        self._write_conf( model, output_name, configuration )

    def write_conf( self,model,output_name, minimization_steps=4800, num_steps = 1e6,
                    output_directory = 'output',
                    update_dimensions=True, extrabonds=True ): 

        """ Write a NAMD configuration file (developed for the mrdna package) """
        
        num_steps = int(num_steps//12)*12
        minimization_steps = int(minimization_steps//24)*24
        if num_steps < 12:
            raise ValueError("Must run with at least 12  steps")
        if minimization_steps < 24:
            raise ValueError("Must run with at least 24 minimization steps")

        format_data = model.__dict__.copy() # get parameters from System object

        format_data['extrabonds'] = """extraBonds on
extraBondsFile $prefix.exb
""" if extrabonds else ""

        if self.useTclForces:
            format_data['margin'] = ""
            format_data['tcl_forces'] = """tclForces on
tclForcesScript $prefix.forces.tcl
"""
        else:
            format_data['margin'] = """margin              30
"""
            format_data['tcl_forces'] = ""

        if update_dimensions:
            format_data['dimensions'] = model.dimensions_from_structure()

        for k,v in zip('XYZ', format_data['dimensions']):
            format_data['origin'+k] = -v*0.5
            format_data['cell'+k] = v

        format_data['prefix'] = output_name
        format_data['minimization_steps'] = int(minimization_steps//2)
        format_data['num_steps'] = num_steps
        format_data['output_directory'] = output_directory
        filename = '{}.namd'.format(output_name)

        with open(filename,'w') as fh:
            fh.write("""
set prefix {prefix}
set nLast 0;			# increment when continueing a simulation
set n [expr $nLast+1]
set out {output_directory}/$prefix-$n
set temperature {temperature}

structure          $prefix.psf
coordinates        $prefix.pdb

outputName         $out
XSTfile            $out.xst
DCDfile            $out.dcd

#############################################################
## SIMULATION PARAMETERS                                   ##
#############################################################

# Input
paraTypeCharmm	    on
parameters          charmm36.nbfix/par_all36_na.prm
parameters	    charmm36.nbfix/par_water_ions_na.prm

wrapAll             off

# Force-Field Parameters
exclude             scaled1-4
1-4scaling          1.0
switching           on
switchdist           8
cutoff              10
pairlistdist        12
{margin}

# Integrator Parameters
timestep            2.0  ;# 2fs/step
rigidBonds          all  ;# needed for 2fs steps
nonbondedFreq       1
fullElectFrequency  3
stepspercycle       12

# PME (for full-system periodic electrostatics)
PME                 no
PMEGridSpacing      1.2

# Constant Temperature Control
langevin            on    ;# do langevin dynamics
# langevinDamping     1   ;# damping coefficient (gamma); used in original study
langevinDamping     0.1   ;# less friction for faster relaxation
langevinTemp        $temperature
langevinHydrogen    off    ;# don't couple langevin bath to hydrogens

# output
useGroupPressure    yes
xstFreq             4800
outputEnergies      4800
dcdfreq             4800
restartfreq         48000

#############################################################
## EXTRA FORCES                                            ##
#############################################################

# ENM and intrahelical extrabonds
{extrabonds}
{tcl_forces}

#############################################################
## RUN                                                     ##
#############################################################

# Continuing a job from the restart files
cellBasisVector1 {cellX} 0 0
cellBasisVector2 0 {cellY} 0
cellBasisVector3 0 0 {cellZ}

if {{$nLast == 0}} {{
    temperature 300
    fixedAtoms on
    fixedAtomsForces on
    fixedAtomsFile $prefix.fixed.pdb
    fixedAtomsCol B
    minimize {minimization_steps:d}
    fixedAtoms off
    minimize {minimization_steps:d}
}} else {{
    bincoordinates  {output_directory}/$prefix-$nLast.restart.coor
    binvelocities   {output_directory}/$prefix-$nLast.restart.vel
}}

run {num_steps:d}
""".format(**format_data))


""" External Runners
Integration with HydroPro and APBS for shape rigid body calculations.
Original script by Chun Kit Chan, 2024
HydroProRunner and APBSRunner module provides interfaces to external tools used in shape rigid body modeling:
- HydroPro for hydrodynamic calculations
- APBS for electrostatics calculations
"""

class HydroProRunner:
    """Interface to HydroPro for hydrodynamic calculations"""
    def __init__(self, mass, simconf=None, structure_name="hydrocal", cal_type="shape"):
        """Initialize HydroPro interface.
        
        Args:
            mass: Mass in AMU
            simconf: SimConf object with temperature, viscosity and solvent_density (optional)
            structure_name: Base name of structure files
            cal_type: shape or mesh, determined by program
        """
        if simconf is None:
            from . import DefaultSimConf
            simconf = DefaultSimConf()
            
        self.temperature = simconf.temperature
        self.viscosity = simconf.viscosity
        self.solvent_density = simconf.solvent_density
        self.binary = simconf.get_binary('hydropro')
            
        # Check if binary exists
        if self.binary is None:
            raise FileNotFoundError("HydroPro binary not found. Please configure in simconf.")
        
        self.binary = Path(self.binary)
        if not self.binary.exists():
            raise FileNotFoundError(f"HydroPro binary not found at {self.binary}")
            
        # Make binary executable if needed (Unix only)
        if platform.system() != 'Windows' and not os.access(self.binary, os.X_OK):
            os.chmod(self.binary, 0o755)
            
        self.structure_name = structure_name[0:10] if len(structure_name)>10 else structure_name

        self.full_name =structure_name
        self.mass = mass
        self.cal_type = cal_type
        
    def write_config(self, output_path="hydropro.dat",
                     aer=2.9,nsig=6,sig_min=1,sig_max=2,specific_volume=0.702,):
        """Write HydroPro configuration file with explicit parameters.
        
        Args:
            output_path: Path to write config file
            cal_type: shape(0) or mesh(1)
            structure_name: Name of the molecule/structure
            mass: Molecular weight in Daltons (amu)
            aer: Atomic Element Radius in Angstroms
            nsig: Number of values of the shell thickness
            sig_min: Minimum radius of beads in the shell (Angstroms)
            sig_max: Maximum radius of beads in the shell (Angstroms)
            specific_volume: Partial specific volume in cm³/g
        """
        temperature_c = self.temperature - 273.15  # Convert K to C
        if self.cal_type=="mesh" or self.cal_type==1:
            aer=10
            nsig=4
            sig_min=10
            sig_max=20
            specific_volume=1

        with open(output_path, 'w') as f:
            # Basic identification
            f.write(f"{self.structure_name}\n")                  # Name of molecule
            f.write(f"{self.structure_name[0:10]}.hydro\n")           # Base name for output files
            f.write("hydro.pdb\n")                         
            f.write("1\n")                                  # Calculation type always 1 (bead surface model)

            # Bead model parameters
            f.write(f"{aer},\n")                            # AER (radius in Angstroms)
            f.write(f"{nsig},\n")                           # NSIG (values of shell thickness)
            f.write(f"{sig_min},\n")                        # SIGMIN (min bead radius)
            f.write(f"{sig_max},\n")                        # SIGMAX (max bead radius)
            
            # Physical parameters
            f.write(f"{temperature_c},\n")                  # Temperature in Celsius
            f.write(f"{self.viscosity},\n")                      # Solvent viscosity in poise
            f.write(f"{self.mass},\n")                           # Molecular weight in Daltons
            f.write(f"{specific_volume},\n")                # Partial specific volume
            f.write(f"{self.solvent_density}\n")                 # Solvent density
            
            # Calculation control parameters
            f.write("-1\n")                       # Number of Q values
            f.write("-1\n")                      # Number of intervals
            f.write("0\n")                      # Monte Carlo trials
            f.write("1\n")                                  # IDIF=1 (yes) for full diffusion tensors
            f.write("*")                                    # End marker


    def parse_output(self, output_file):
        mass=self.mass
        """Parse HydroPro output file to get damping coefficients.
        
        Args:
            output_file: Path to HydroPro output file
            mass: Mass in AMU used to normalize coefficients
            
        Returns:
            tuple of (translation_damping, rotation_damping)
        """
        with open(output_file) as f:
            lines = f.readlines()
        mass=self.mass
            
        # Skip header
        line_num = 48
        
        # Read translational coefficients
        Dx = float(lines[line_num].strip().split()[0])
        Dy = float(lines[line_num+1].strip().split()[1])
        Dz = float(lines[line_num+2].strip().split()[2])
        
        # Skip two lines
        line_num += 5
        
        # Read rotational coefficients
        Rx = float(lines[line_num].strip().split()[3])
        Ry = float(lines[line_num+1].strip().split()[4])
        Rz = float(lines[line_num+2].strip().split()[5])
        
        # Convert units
        # Translation: "(295 k K) / (( cm^2/s) *  amu)" "1/ns"
        trans_damp = [24.527692/(x*mass) for x in [Dx, Dy, Dz]]
        
        # Rotation: "(295 k K) / ((1 /s) *  amu AA^2)" "1/ns"
        rot_damp = [2.4527692e+17 / (x*mass) for x in [Rx, Ry, Rz]]
        
        return trans_damp, rot_damp
                
    def run_calculation(self,work_dir="."):
        """Run HydroPro calculation.
        
        Args:
            structure_name: Base name of structure files
            mass: Mass in AMU
            work_dir: Working directory for calculation
        
        Returns:
            Dictionary containing:
            - translation_damping: [Dx, Dy, Dz]
            - rotation_damping: [Rx, Ry, Rz]
        """
        structure_name, mass=self.structure_name, self.mass
        original_dir = os.getcwd()
        try:
            os.chdir(work_dir)
            
            # Write config
            self.write_config()
            
            # Link structure file
            pdb_path = Path(f"{self.full_name}.pdb")
            if not pdb_path.exists():
                raise FileNotFoundError(f"Structure file not found: {pdb_path}")
            
            os.symlink(pdb_path, "hydro.pdb")
            # Run HydroPro
            result = subprocess.run(str(self.binary), capture_output=True, 
                                 text=True,check=True)
            

            trans_damp, rot_damp = self.parse_output(f"{structure_name}.hydro-res.txt")
            
            return trans_damp, rot_damp
            
            
        finally:
            if os.path.exists("hydro.pdb"):
                os.unlink("hydro.pdb")
            os.chdir(original_dir)

class APBSRunner:
    """Interface to APBS for electrostatics calculations.
    
    This class provides an interface to run APBS (Adaptive Poisson-Boltzmann Solver)
    for calculating electrostatic properties of molecular systems.
    """
    
    def __init__( self, structure_name, xyz_dims, buffer: float = 50, large_system: bool = False,
        dividend: int = 2,simconf= None,binary_path= None,psize_script= None,):
        """Initialize APBS interface.
        
        Args:
            structure_name: Base name of structure files
            xyz_dims: [x, y, z] dimensions
            buffer: Grid buffer size in Å
            large_system: Whether to use large system mode
            dividend: Grid density reduction factor for large system
            simconf: SimConf object (optional)
            binary_path: Path to APBS executable. If None, uses path from simconf
            psize_script: Optional path to psize script
            
        Raises:
            FileNotFoundError: If APBS binary cannot be found
            ValueError: If invalid parameters are provided
        """
        self.structure_name = structure_name
        self.xyz_dims = xyz_dims
        self.buffer = float(buffer)
        self.large_system = bool(large_system)
        self.dividend = int(dividend)
        
        if not xyz_dims or len(xyz_dims) != 3:
            raise ValueError("xyz_dims must be a list of 3 dimensions")
        
        # Get binary path using priority order
        self.binary = Path(simconf.get_binary('apbs'))
        self.psize = Path(psize_script) if psize_script else None
        
        # Store simulation configuration parameters
        if simconf is not None:
            self.temperature = simconf.temperature
            self.viscosity = simconf.viscosity
            self.solvent_density = simconf.solvent_density
            self.salt = simconf.salt
        else:
            # Default values if simconf not provided
            self.temperature = 298.15  # K
            self.viscosity = 0.89     # centipoise
            self.solvent_density = 1.0 # g/cm^3
            self.salt = 0.15          # M


    def write_config(self) -> None:
        """Write APBS configuration file.
        
        Creates an APBS input file with the current configuration parameters.
        The file is named {structure_name}.apbs
        
        Raises:
            IOError: If unable to write configuration file
        """
        # Calculate grid dimensions
        xyz_cg = [str(int(dim + self.buffer)) for dim in self.xyz_dims]
        
        if not self.large_system:
            xyz_dime = xyz_cg
        else:
            # For large systems, reduce grid density
            xyz_dime = [str(int((dim + self.buffer) / self.dividend)) for dim in self.xyz_dims]
            
        center = 'mol 1'  # Use molecule center for both cases
        
        config = f"""read
mol pqr {self.structure_name}.pqr
end
elec
mg-auto
dime {' '.join(xyz_dime)}
cglen {' '.join(xyz_cg)}
cgcent {center}
fglen {' '.join(xyz_cg)}
fgcent {center}
mol 1
npbe
bcfl sdh
srfm smol
chgm spl2
ion 1 {self.salt} 2.0
ion -1 {self.salt} 2.0
pdie 12.0
sdie 78.54
sdens 10.0
srad 1.4
swin 0.3
temp {self.temperature}
gamma 0.105
calcenergy no
calcforce no
write pot dx {self.structure_name}.elec
end
quit"""

        try:
            with open(f"{self.structure_name}.apbs", 'w') as f:
                f.write(config)
        except IOError as e:
            raise IOError(f"Failed to write APBS configuration file: {e}")
            
    def run_calculation(self, workdir=".") -> Path:
        """Run APBS calculation.
        
        Args:
            workdir: Working directory for the calculation
            
        Returns:
            Path: Path to output potential file
            
        Raises:
            RuntimeError: If APBS calculation fails
            FileNotFoundError: If output file is not generated
        """
        workdir = Path(workdir)
        original_dir = Path.cwd()
        
        try:
            os.chdir(workdir)
            
            # Write configuration file
            self.write_config()
            
            # Run APBS
            try:
                result = subprocess.run(
                    [str(self.binary), f"{self.structure_name}.apbs"],
                    capture_output=True,
                    text=True,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"APBS calculation failed: {e.stderr}")
            
            # Check for output file
            output_file = Path(f"{self.structure_name}.elec.dx")
            if not output_file.exists():
                raise FileNotFoundError("APBS failed to generate output file")
                
            return output_file
            
        finally:
            os.chdir(original_dir)

class TclScriptGenerator:
    """
    Generator for TCL scripts used in structure processing and visualization
    """
    
    def __init__(self, work_dir=None):
        """Initialize with working directory path"""
        work_dir = Path(work_dir) if work_dir else Path.cwd()
        self.work_dir=work_dir.absolute()


    def write_align_tcl(self, filename="align.tcl"):
        """
        Write the alignment TCL script to file
        
        Args:
            filename: Name of the output TCL script
        
        Returns:
            Path to the written script
        """
        script_path = self.work_dir / filename
        script_content = '''lassign $argv prefix
set seltext all

proc rotationIsRightHanded {R {tol 0.01}} {
    set x [coordtrans $R {1 0 0}]
    set y [coordtrans $R {0 1 0}]
    set z [coordtrans $R {0 0 1}]

    set l [veclength [vecsub $z [veccross $x $y]]]
    return [expr {$l < $tol}]
}

set ID [mol new $prefix.psf]
mol addfile $prefix.pdb waitfor all
set all [atomselect $ID all]
set sel [atomselect $ID $seltext]

## Center system on $sel
$all moveby [vecinvert [measure center $sel weight mass]]

set continue 1
while { $continue } {
    ## Get current moment of inertia to determine rotation to align
    lassign [measure inertia $sel moments] com principleAxes

    ## Convert 3x3 rotation to 4x4 vmd transformation
    set R [trans_from_rotate $principleAxes]
    ## Fix left-handed principle axes sometimes returned by 'measure inertia'
    if { ! [rotationIsRightHanded $R] } {
        puts "This was true"
        # puts "rotation $R is not right handed! Fixing!"
        set R [transmult {{1 0 0 0} {0 1 0 0} {0 0 -1 0} {0 0 0 1}} $R]
    }

    puts "My rotation is here: $R"
    puts "My second line is here: [lassign [measure inertia $sel moments] com principleAxes]"

    ## Apply rotation and check that it worked
    $all move $R

    ## Get current moment of inertia to determine rotation to align

    lassign [measure inertia $sel moments] com principleAxes moments
    puts $principleAxes
    set goodcount 0
    foreach x0 {{1 0 0} {0 1 0} {0 0 1}} {
        set x [coordtrans [trans_from_rotate $principleAxes] $x0]
        if {[veclength [vecsub $x $x0]] < 0.01} {
            incr goodcount
        }
    }
    if { $goodcount == 3 } {
        set continue 0
    }
}

## Write transformation matrix to return to original conformation
set ch [open $prefix.rotate-back.txt w]
foreach line [trans_to_rotate [transtranspose $R]] {
    puts $ch $line
}
close $ch

## Write out moments of inertia
set ms ""
foreach m $moments { lappend ms [veclength $m] }
set ch [open $prefix.inertia.txt w]
puts $ch $ms
close $ch

## Write out mass
set ch [open $prefix.mass.txt w]
puts $ch [measure sumweights $sel weight mass]
close $ch

## Write out psf, pdb of transformed selection
$sel writepdb $prefix.aligned.pdb
$sel writepsf $prefix.aligned.psf'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        logger.debug(f"Wrote alignment script to {script_path}")
        return script_path
    
    def write_charge_density_tcl(self, filename="charge-density.tcl", resolution=2.0):
        """
        Write the alignment TCL script to file
        
        Args:
            filename: Name of the output TCL script
        
        Returns:
            Path to the written script
        """
        charge_tcl = self.work_dir / filename
        with open(charge_tcl, 'w') as f:
            f.write(f'''lassign $argv prefix
set resolution {resolution}
set ID [mol new $prefix.psf]
mol addfile $prefix.pdb
set all [atomselect $ID all]
set netCharge [measure sumweights $all weight charge]

## Write out charge density
volmap density $all -o $prefix.chargeDensity.dx -res $resolution -weight charge
## Write out pqr for subsequent EM calculation
$all writepqr $prefix.pqr

set ch [open $prefix.netCharge.dat w]
puts $ch $netCharge
close $ch

set ch [open $prefix.dimension.dat w]
set minmax [measure minmax $all]
set x_dim [expr [lindex [lindex $minmax 1] 0] - [lindex [lindex $minmax 0] 0]]
set y_dim [expr [lindex [lindex $minmax 1] 1] - [lindex [lindex $minmax 0] 1]]
set z_dim [expr [lindex [lindex $minmax 1] 2] - [lindex [lindex $minmax 0] 2]]
puts $ch $x_dim
puts $ch $y_dim
puts $ch $z_dim
close $ch''')


    def process_vdw_parameters(self, protein_types):
        """
        Process VDW parameters from protein types

        Args:
            protein_types: List of protein particle types

        Returns:
            Dictionary containing VDW parameters
        """
        vdw_params = {}
        for ptype in protein_types:
            if hasattr(ptype, 'vdw_params'):
                params = ptype.vdw_params
                vdw_params[ptype.name] = {
                    'radius': params.get('radius', 1.0),
                    'epsilon': params.get('epsilon', 1.0)
                }
        return vdw_params

    def write_vdw_tcl(self, filename="vdw.tcl"):
        """
        Write the VDW TCL script to file
        
        Args:
            filename: Name of the output TCL script
        
        Returns:
            Path to the written script
        """
        script_path = self.work_dir / filename
        script_content = self.generate_vdw_script()
        
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        logger.debug(f"Wrote VDW script to {script_path}")
        return script_path
    
    def generate_vdw_maps_diffusive(self, potResolution=1, denResolution=2,num_heavy_cluster=3):
        """Generate VDW maps using VMD."""
        aligned_name = f"{self.base_name}.aligned"
        aligned_path = str(self.work_dir / aligned_name)
        
        # Create VDW TCL script
        vdw_tcl = self.work_dir / "vdw.tcl"
        with open(vdw_tcl, 'w') as f:
            f.write(f'''set prefixes $argv
set potResolution ''' + str(potResolution) + '''
set denResolution ''' + str(denResolution) + '''
set IDs ""
set minRadius 0.5

package require ilstools

ILStools::readcharmmparams [glob ''' + self.work_dir + '''/*]

set ljParms ""
set lj_hyd ""

foreach prefix $prefixes {
    set ID [mol new $prefix.psf]
    mol addfile $prefix.pdb
    lappend IDs $ID

    ILStools::assigncharmmparams $ID; # sets radius and occupancy to rmin and eps

    set all [atomselect $ID "noh"]
    set hyd [atomselect $ID "hydrogen"]
    append ljParms " [lsort -unique [$all get {radius occupancy}]]"
    append lj_hyd " [lsort -unique [$hyd get {radius occupancy}]]"
}

set ljParms [lsort -unique $ljParms]
set lj_hyd [lsort -unique $lj_hyd]
set ljTypes ""

set ch [open tmp.dat w]
foreach vals $ljParms {
    lassign $vals r e
    if {$r < $minRadius} {continue};

    set count 0
    set types ""
    foreach ID $IDs {
        set sel [atomselect $ID "noh and radius $r and occupancy \\"$e\\""]
        incr count [$sel num]
        append types " [lsort -unique [$sel get type]]"
    }
    set types [lsort -unique $types]
    lappend ljTypes $types
    puts $ch "$r $e $count"
}
close $ch

set ch [open hyd.dat w]
foreach vals $lj_hyd {
    lassign $vals r e
    if {$r < $minRadius} {continue};

    set count 0
    set types ""
    foreach ID $IDs {
        set sel [atomselect $ID "hydrogen and radius $r and occupancy \\"$e\\""]
        incr count [$sel num]
        append types " [lsort -unique [$sel get type]]"
    }
    set types [lsort -unique $types]
    lappend ljTypes $types
    puts $ch "$r $e $count"
}
close $ch
puts "My LJ types are: $ljTypes"
''')