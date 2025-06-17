import numpy as np
import parmed
from scipy.spatial import distance_matrix
from .model import ArbdModel
from .core_objects import Group, ParticleType, PointParticle
from .interactions import HarmonicBond
from .logger import logger


class ParmedArbd(ArbdModel):
    """
    Class for converting ParmEd structures to ARBD models for simulation.
    
    This class facilitates the conversion of molecular structures loaded with ParmEd
    into ARBD simulation models, preserving bonded and non-bonded interactions.
    It also supports creating dual topology models for free energy calculations.
    """
    
    def __init__(self, parmed_structure=None, psf=None, pdb=None, parameter_files=None, 
                 system_type='charmm', integrator='MD', cutoff=12, **kwargs):
        """
        Initialize ParmedArbd model from a ParmEd structure or PSF/PDB files.
        
        Args:
            parmed_structure: Existing ParmEd structure object (optional)
            psf: Path to PSF file (optional if parmed_structure is provided)
            pdb: Path to PDB file (optional if parmed_structure is provided)
            parameter_files: List of parameter files for force field (optional)
            system_type: Type of system ('charmm', etc.)
            integrator: Simulation integrator type
            cutoff: Non-bonded interaction cutoff distance
            **kwargs: Additional arguments for ArbdModel

        Attributes:
            atom_types: Dictionary mapping atom type names to ParticleType objects
            atoms_map: Mapping from ParmEd atoms to ARBD atoms
        """
        # Initialize empty model first
        ArbdModel.__init__(self, [], integrator=integrator, cutoff=cutoff, **kwargs)
        
        self.atom_types = {}
        self.atoms_map = {}
        
        # Load structure if provided
        if parmed_structure is not None:
            self.parmed_structure = parmed_structure
        elif psf is not None and pdb is not None:
            self.parmed_structure = self._read_files(psf, pdb, parameter_files, system_type)
        else:
            self.parmed_structure = None
            logger.warning("No structure provided; use load_structure() to load one")
            return
            
        # Convert the structure to ARBD model
        if self.parmed_structure is not None:
            self._build_model_from_structure()
    
    def _read_files(self, psf, pdb, parameter_files=None, system_type='charmm'):
        """
        Read PSF and PDB files into a ParmEd structure.
        
        Args:
            psf: Path to PSF file
            pdb: Path to PDB file
            parameter_files: List of parameter files
            system_type: Type of system ('charmm', etc.)
            
        Returns:
            ParmEd structure object
        """
        if system_type == 'charmm':
            p1 = parmed.charmm.CharmmPsfFile(psf)
            c1 = parmed.load_file(pdb)
            for a, b in zip(p1.atoms, c1.atoms):
                a.xx = b.xx 
                a.xy = b.xy
                a.xz = b.xz
                a.bfactor = b.bfactor
                
            if parameter_files is not None:
                if isinstance(parameter_files, str):
                    parameter_files = [parameter_files]
                params = parmed.charmm.CharmmParameterSet(*parameter_files)
                p1.load_parameters(params)
                
        else:
            raise NotImplementedError(f'Cannot import a "{system_type}" model')
            
        return p1
        
    def load_structure(self, parmed_structure=None, psf=None, pdb=None, 
                       parameter_files=None, system_type='charmm'):
        """
        Load a ParmEd structure or read from PSF/PDB files.
        
        Args:
            parmed_structure: Existing ParmEd structure object (optional)
            psf: Path to PSF file (optional if parmed_structure is provided)
            pdb: Path to PDB file (optional if parmed_structure is provided)
            parameter_files: List of parameter files (optional)
            system_type: Type of system ('charmm', etc.)
            
        Returns:
            self for method chaining
        """
        if parmed_structure is not None:
            self.parmed_structure = parmed_structure
        elif psf is not None and pdb is not None:
            self.parmed_structure = self._read_files(psf, pdb, parameter_files, system_type)
        else:
            raise ValueError("Either parmed_structure or both psf and pdb must be provided")
            
        # Clear existing model and rebuild
        self.clear_all()
        self._build_model_from_structure()
        
        return self
    
    def _validate_atom(self, atom):
        """
        Validate that atom properties are compatible with ARBD model.
        
        Args:
            atom: ParmEd atom object to validate
            
        Raises:
            NotImplementedError: If atom has unsupported properties
        """
        try:
            atom.multipoles
            raise NotImplementedError(f'Atom {atom} uses multipoles attribute for AMEOBA')
        except AttributeError:
            pass

        for k in 'tortors other_locations anisou hybridization irotat tree screen solvent_radius join altloc marked cmaps children aromatic formal_charge'.split():
            _a = atom.__getattribute__(k)
            if _a is not None:
                _isnum = isinstance(_a, float) or isinstance(_a, int)
                if (_isnum and (_a != 0)) or ((not _isnum) and len(_a) > 0):
                    _msg = f'Atom {atom.idx} "{k}" is non-null ({_a})'
                    logger.warning(_msg)
        
        if len(atom.other_locations) > 0:
            _msg = f'Atom {atom.idx} has other locations which is not supported'
            raise NotImplementedError(_msg)
    
    def _validate_topology(self):
        """
        Validate that the topology is compatible with ARBD model.
        
        Raises:
            NotImplementedError: If topology has unsupported features
        """
        part = self.parmed_structure
        
        if len(part.rb_torsions) > 0:
            _msg = f'Topology includes RB torsions'
            raise NotImplementedError(_msg)
            
        if len(part.urey_bradleys) > 0:
            _msg = f'Topology includes Urey Bradley terms'
            logger.warning(_msg)
            
        for k in ['impropers', 'cmaps', 'trigonal_angles', 'out_of_plane_bends',
                 'pi_torsions', 'stretch_bends', 'torsion_torsions', 'chiral_frames',
                 'multipole_frames', 'adjusts', 'links']:
            _a = part.__getattribute__(k)
            if len(_a) > 0:
                _msg = f'Topology includes {len(_a)} "{k}"'
                logger.warning(_msg)

        if part.unknown_functional != False:
            _msg = f'Topology unknown_functional is "{part.unknown_functional}" (not "False")'
            raise NotImplementedError(_msg)

        if part.symmetry is not None:
            _msg = f'Topology symmetry is "{part.symmetry}" (not "None")'
            raise NotImplementedError(_msg)

        if part.nrexcl > 4:
            _msg = f'Topology nrexcl is "{part.nrexcl}" (not <= "4")'
            raise NotImplementedError(_msg)
            
        if part._combining_rule != 'lorentz':
            _msg = f'Unrecognized non-bonded combining rule "{part._combining_rule}"'
            raise NotImplementedError(_msg)
        
        # Validate first atom as representative
        self._validate_atom(part.atoms[0])
    
    def _create_particle_types(self):
        """
        Create ARBD ParticleType objects from atom types in the ParmEd structure.
        
        Returns:
            Dictionary mapping atom type names to ParticleType objects
        """
        _atypes = set(a.atom_type for a in self.parmed_structure.atoms)
        logger.info(f'Importing {len(_atypes)} types from ParmEd')
        
        _new_types_d = {}
        for t in _atypes:
            _kwargs = {k: t.__getattribute__(k)
                      for k in 'number mass atomic_number epsilon rmin charge'.split()}

            for k in 'epsilon_14 rmin_14'.split():
                _a1 = t.__getattribute__(k)
                _a2 = t.__getattribute__(k.split('_')[0])
                if _a1 != _a2:
                    _msg = f'{t}.{k} differs from normal term ({_a1} != {_a2})'
                    logger.warning(_msg)

            for k in 'nbfix nbthole _bond_type'.split():
                _a = t.__getattribute__(k)
                if _a is not None and len(_a) > 0:
                    _msg = f'{t}.{k} is non-null ({_a})'
                    logger.warning(_msg)

            assert(t.name not in _new_types_d)
            _new_types_d[t.name] = ParticleType(t.name,
                                               damping_coefficient=1e-4,  # 0.1 ps
                                               **_kwargs)

        self.atom_types = _new_types_d
        return _new_types_d
    
    def _build_model_from_structure(self):
        """
        Build ARBD model from the loaded ParmEd structure.
        """
        # Validate topology first
        self._validate_topology()
        
        # Create particle types
        atom_types = self._create_particle_types()
        
        # Create model structure
        allsegs = Group(name='allsegs', _segments={})
        atoms_d = {}
        bond_t_d = {}
        
        # Helper function to get or create residue group
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
        
        # Helper function to create an atom
        def _add_atom(atom):
            a = atom
            mapping = dict(bfactor='beta',
                          occupancy=None)
            for k, v in list(mapping.items()):
                if v is None:
                    mapping[k] = k

            res = _get_residue(a.residue)
            _a = PointParticle(name=a.name, type_=atom_types[a.type],
                              position=np.array((a.xx, a.xy, a.xz)), parent=res,
                              **{mapping[k]: a.__getattribute__(k) for k in mapping.keys()})
            atoms_d[a] = _a
            return _a
        
        # Helper function to get or create a bond type
        def _get_bond_type(bond):
            t = bond.type
            if t not in bond_t_d:
                if t.penalty is not None:
                    logger.warning(f'Bond.penalty for bond {t} not null')
                width = 0.76565392/np.sqrt(np.abs(t.k))  # units conversion
                b = HarmonicBond(k=t.k, r0=t.req,
                                resolution=0.02*width,
                                range_=[max(0, t.req-5*width), t.req+5*width])
                bond_t_d[t] = b
            return bond_t_d[t]
        
        # Add atoms to model
        logger.info(f'Importing {len(self.parmed_structure.atoms)} atoms from ParmEd')
        for i, a in enumerate(self.parmed_structure.atoms):
            assert(a.idx == i)
            _add_atom(a)
        
        # Add bonds to model
        logger.info(f'Importing {len(self.parmed_structure.bonds)} bonds from ParmEd')
        for b0 in self.parmed_structure.bonds:
            t = _get_bond_type(b0)
            a1, a2 = [atoms_d[a] for a in (b0.atom1, b0.atom2)]
            seg = a1.parent.parent
            seg.add_bond(a1, a2, t, self.parmed_structure.nrexcl >= 1)
        
        # Add the segment group to the model
        self.add(allsegs)
        
        # Store mapping from ParmEd atoms to ARBD atoms
        self.atoms_map = atoms_d
        
        return self
    
    def simulate(self, output_name, output_directory='output', log_file=None,
                directory='.', binary=None, num_procs=None, dry_run=False, **conf_params):
        """
        Run simulation with the ARBD model.
        
        Args:
            output_name: Base name for output files
            output_directory: Directory for output files
            log_file: File for logging
            directory: Working directory
            binary: Path to simulation binary
            num_procs: Number of processors to use
            dry_run: If True, don't actually run
            **conf_params: Additional configuration parameters
            
        Returns:
            Result of the simulation
        """
        from .engine import ArbdEngine
        
        engine = ArbdEngine(timestep=conf_params.get('timestep', 1e-6))
        
        return engine.simulate(
            self, output_name, 
            output_directory=output_directory,
            directory=directory,
            log_file=log_file,
            binary=binary,
            num_procs=num_procs,
            dry_run=dry_run,
            **conf_params
        )
    
