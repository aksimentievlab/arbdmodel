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
    
    Attributes:
        parmed_structure: The original ParmEd structure
        atom_types: Dictionary mapping atom type names to ParticleType objects
        atoms_map: Mapping from ParmEd atoms to ARBD atoms
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
    
    @classmethod
    def create_dual_topology_model(cls, p1, p2, u1=None, u2=None, **kwargs):
        """
        Create a dual topology ARBD model from two ParmEd structures.
        
        Args:
            p1: First ParmEd structure
            p2: Second ParmEd structure
            u1: Optional MDAnalysis universe for p1
            u2: Optional MDAnalysis universe for p2
            **kwargs: Additional arguments for ParmedArbd
            
        Returns:
            ParmedArbd model with dual topology
        """
        # Create the dual topology
        dual_structure, pairs, bp_pairs = cls.create_dual_topology(p1, p2, u1, u2)
        
        # Create ParmedArbd model with the dual structure
        model = cls(parmed_structure=dual_structure, **kwargs)
        
        # Store the extra bond information for use in simulation
        model.dual_topology_pairs = pairs
        model.bp_pairs = bp_pairs
        
        return model

    def write_restraint_files(self, fep_file='dual.fep.exb', min_fep_file='dual.min.fep.exb', 
                            bp_file='dual.bp.exb'):
        """
        Write restraint files for dual topology simulations.
        
        Args:
            fep_file: Output file for FEP restraints
            min_fep_file: Output file for minimization FEP restraints
            bp_file: Output file for base pair restraints
        """
        if not hasattr(self, 'dual_topology_pairs'):
            raise AttributeError("This model doesn't have dual topology information. "
                                "Use create_dual_topology_model to create one.")
        
        with open(fep_file, 'w') as fh:
            with open(min_fep_file, 'w') as fh2:
                for i, j, r0 in self.dual_topology_pairs:
                    fh.write(f'bond {i} {j} 1 {r0}\n')
                    fh2.write(f'bond {i} {j} 100 {r0}\n')
        
        if hasattr(self, 'bp_pairs') and self.bp_pairs:
            with open(bp_file, 'w') as fh:
                for i, j in self.bp_pairs:
                    fh.write(f'bond {i} {j} 1 2.8\n')

    @staticmethod
    def convert_sod_to_mg(structure):
        """
        Convert sodium ions to magnesium ions in a structure.
        
        Args:
            structure: ParmEd structure to modify
            
        Returns:
            Modified structure
        """
        for a in structure:
            if a.residue.name == 'WAT':
                break
            elif a.residue.name != 'Na+':
                continue
            
            # Find appropriate magnesium atom type
            try:
                mg_type = structure.atom_types['Mg2+']
            except KeyError:
                # If Mg2+ atom type doesn't exist, create it based on Na+ properties
                na_type = a.atom_type
                mg_type = parmed.AtomType('Mg2+', 12, 24.305, 2.0)
                mg_type.epsilon = na_type.epsilon
                mg_type.rmin = na_type.rmin
                mg_type.epsilon_14 = na_type.epsilon_14
                mg_type.rmin_14 = na_type.rmin_14
                structure.atom_types[mg_type.name] = mg_type
            
            # Update atom properties to magnesium
            a.atom_type = mg_type
            a.residue.name = 'Mg2+'
            a.name = 'Mg2+'
            a.type = 'Mg2+'
            a.atomic_number = 12
            a._charge = 2
            a.charge = 2
            a.mass = 24.305
        
        return structure
    
    @staticmethod
    def create_dual_topology(p1, p2, u1=None, u2=None):
        """
        Create a dual topology structure from two ParmEd structures.
        
        Args:
            p1: First ParmEd structure
            p2: Second ParmEd structure
            u1: Optional MDAnalysis universe for p1
            u2: Optional MDAnalysis universe for p2
            
        Returns:
            Tuple of (combined structure, restraint pairs, base pair bonds)
        """
        # Ensure structures have same number of residues
        assert len(p1.residues) == len(p2.residues)
        
        # Check index alignment
        i = p1.residues[5].atoms[0].idx
        assert i == p2.residues[5].atoms[0].idx
        
        # Verify compatibility
        assert len(p2.urey_bradleys) == 0
        assert len(p2.cmaps) == 0
        
        # Create new empty structure with same parameters as p1
        new = p1[:0]
        
        # Mapping dictionaries
        p2_to_p1 = dict()
        u_p2_to_new = dict()
        
        # Create maps for bonded terms
        bond_map = {a: [] for a in p2.atoms}
        angle_map = {a: [] for a in p2.atoms}
        dihed_map = {a: [] for a in p2.atoms}
        improper_map = {a: [] for a in p2.atoms}
        
        # Populate bonded term maps
        for b in p2.bonds:
            bond_map[b.atom1].append((b, 0))
            bond_map[b.atom2].append((b, 1))
        for b in p2.angles:
            angle_map[b.atom1].append((b, 0))
            angle_map[b.atom2].append((b, 1))
            angle_map[b.atom3].append((b, 2))
        for b in p2.dihedrals:
            dihed_map[b.atom1].append((b, 0))
            dihed_map[b.atom2].append((b, 1))
            dihed_map[b.atom3].append((b, 2))
            dihed_map[b.atom4].append((b, 3))
        for b in p2.impropers:
            improper_map[b.atom1].append((b, 0))
            improper_map[b.atom2].append((b, 1))
            improper_map[b.atom3].append((b, 2))
            improper_map[b.atom4].append((b, 3))
        
        pairs = []
        
        # Process each segment of p1
        last_p1_i = 0
        first_p1_atom = dict()  # keys are segids
        first_new_atom = dict()  # keys are segids
        last_p1_new_atom = dict()  # keys are segids
        processed_segments = set()
        
        # First pass: Add all atoms from p1 and unique atoms from p2
        for res1, res2 in zip(p1.residues, p2.residues):
            seg = res1.segid
            assert seg == res2.segid
            
            # Add complete segment from p1 to new molecule if not already processed
            if seg not in processed_segments:
                _next = [r.atoms[-1].idx + 1 for r in p1.residues if r.segid == seg][-1]
                first_p1_atom[seg] = last_p1_i
                first_new_atom[seg] = len(new.atoms)
                new = new + p1[last_p1_i:_next]
                last_p1_new_atom[seg] = len(new.atoms)
                last_p1_i = _next
                processed_segments.add(seg)
            
            # If residue is different in p1 and p2, add unique atoms from p2
            if res1.name != res2.name:
                # Perform distance search to match atoms
                c1 = np.array([p1.coordinates[a.idx] for a in res1.atoms])
                c2 = np.array([p2.coordinates[a.idx] for a in res2.atoms])
                dists = distance_matrix(c1, c2)
                
                common_ids_j = set()
                common_atoms_j = set()
                
                # Find common atoms based on name, type, and charge
                for i, j in zip(*np.where(dists < 0.1)):
                    a1, a2 = res1.atoms[i], res2.atoms[j]
                    p2_to_p1[a2] = a1
                    if a1.name[0] == 'H':
                        continue  # Handle hydrogens separately
                    if a1.name == a2.name and (a1.type == a2.type) and (a1.charge == a2.charge):
                        common_ids_j.add(j)
                        common_atoms_j.add(a2)
                
                # Handle hydrogen atoms
                for i, j in zip(*np.where(dists < 0.1)):
                    a1, a2 = res1.atoms[i], res2.atoms[j]
                    if a1.name[0] != 'H':
                        continue  # Only consider hydrogens
                    if a1.name == a2.name and (a1.type == a2.type) and (a1.charge == a2.charge):
                        # Check that parent atoms of hydrogens are already considered common
                        assert len(a1.bonds) == 1 and len(a2.bonds) == 1
                        b1 = [a for a in (a1.bonds[0].atom1, a1.bonds[0].atom2) if a != a1][0]
                        b2 = [a for a in (a2.bonds[0].atom1, a2.bonds[0].atom2) if a != a2][0]
                        if b2 in common_atoms_j:
                            common_ids_j.add(j)
                            common_atoms_j.add(a2)
                
                # Special handling for H2' atoms (fix for topology issues)
                if any(a.name == "H2'" for a in res1.atoms) and any(a.name == "H2'" for a in res2.atoms):
                    a1 = [a for a in res1.atoms if a.name == "H2'"][0]
                    j, a2 = next((i, a) for i, a in enumerate(res2.atoms) if a.name == "H2'")
                    if a2 not in p2_to_p1:
                        p2_to_p1[a2] = a1
                        common_ids_j.add(j)
                        common_atoms_j.add(a2)
                
                # Find unique atoms in res2
                unique_atoms_j = [
                    res2.atoms[j] for j in range(len(res2.atoms))
                    if j not in common_ids_j
                ]
                unique_ids_j = [a2.idx for a2 in unique_atoms_j]
                
                # Add unique atoms from res2 to new structure
                if unique_ids_j:
                    logger.info(f'Adding {len(unique_ids_j)} unique atoms from residue {res2.name}')
                    new = new + p2[unique_ids_j]
        
        # Second pass: Add bonded terms and set up dual topology information
        current_seg = None
        current_new_atom = None
        for res1, res2 in zip(p1.residues, p2.residues):
            seg = res1.segid
            assert seg == res2.segid
            
            if current_seg != seg:
                current_seg = seg
                current_new_atom = last_p1_new_atom[seg]
                p1_to_new_di = first_new_atom[seg] - first_p1_atom[seg]
            
            if res1.name != res2.name:
                # Perform distance search to match atoms
                c1 = np.array([p1.coordinates[a.idx] for a in res1.atoms])
                c2 = np.array([p2.coordinates[a.idx] for a in res2.atoms])
                dists = distance_matrix(c1, c2)
                
                common_ids_i = set()
                common_ids_j = set()
                common_atoms_j = set()
                
                # Find common atoms
                for i, j in zip(*np.where(dists < 0.1)):
                    a1, a2 = res1.atoms[i], res2.atoms[j]
                    p2_to_p1[a2] = a1
                    if a1.name[0] == 'H':
                        continue
                    if a1.name == a2.name and (a1.type == a2.type) and (a1.charge == a2.charge):
                        common_ids_i.add(i)
                        common_ids_j.add(j)
                        common_atoms_j.add(a2)
                
                # Handle hydrogen atoms
                for i, j in zip(*np.where(dists < 0.1)):
                    a1, a2 = res1.atoms[i], res2.atoms[j]
                    if a1.name[0] != 'H':
                        continue
                    if a1.name == a2.name and (a1.type == a2.type) and (a1.charge == a2.charge):
                        assert len(a1.bonds) == 1 and len(a2.bonds) == 1
                        b1 = [a for a in (a1.bonds[0].atom1, a1.bonds[0].atom2) if a != a1][0]
                        b2 = [a for a in (a2.bonds[0].atom1, a2.bonds[0].atom2) if a != a2][0]
                        if b2 in common_atoms_j:
                            common_ids_i.add(i)
                            common_ids_j.add(j)
                            common_atoms_j.add(a2)
                
                # Special handling for H2' atoms
                if any(a.name == "H2'" for a in res1.atoms) and any(a.name == "H2'" for a in res2.atoms):
                    i, a1 = next((i, a) for i, a in enumerate(res1.atoms) if a.name == "H2'")
                    j, a2 = next((i, a) for i, a in enumerate(res2.atoms) if a.name == "H2'")
                    if a2 in p2_to_p1:
                        assert p2_to_p1[a2] == a1
                    p2_to_p1[a2] = a1
                    common_ids_i.add(i)
                    common_ids_j.add(j)
                    common_atoms_j.add(a2)
                
                # Get unique atoms in res2
                unique_atoms_j = [
                    res2.atoms[j] for j in range(len(res2.atoms))
                    if j not in common_ids_j
                ]
                unique_ids_j = [a2.idx for a2 in unique_atoms_j]
                
                # Find mapping of unique atoms from res2 to new structure
                _num_new = len(unique_ids_j)
                _tmp = {a.name: a for a in new.atoms[current_new_atom:current_new_atom + _num_new]}
                current_new_atom += _num_new
                u_p2_to_new.update({a2: _tmp[a2.name] for a2 in unique_atoms_j})
                
                # Find pairs of dual-topology atoms for restraints, excluding hydrogens
                for i in range(len(c1)):
                    if i in common_ids_i or res1.atoms[i].name[0] == 'H':
                        dists[i, :] += 100  # Exclude from pairing
                
                for j in range(len(c2)):
                    if j in common_ids_j or res2.atoms[j].name[0] == 'H':
                        dists[:, j] += 100  # Exclude from pairing
                
                # Create restraint pairs
                if len(c1) < len(c2):
                    for i, a1 in enumerate(res1.atoms):
                        if i in common_ids_i or a1.name[0] == 'H':
                            continue
                        j = np.argmin(dists[i])
                        pairs.append((p1_to_new_di + a1.idx, u_p2_to_new[res2.atoms[j]].idx, dists[i, j]))
                else:
                    for j, a2 in enumerate(res2.atoms):
                        if j in common_ids_j or a2.name[0] == 'H':
                            continue
                        i = np.argmin(dists[:, j])
                        pairs.append((p1_to_new_di + res1.atoms[i].idx, u_p2_to_new[a2].idx, dists[i, j]))
                
                # Set beta factor for FEP atoms
                for i, a in enumerate(new.atoms[p1_to_new_di + res1.atoms[0].idx:p1_to_new_di + res1.atoms[-1].idx + 1]):
                    if i not in common_ids_i:
                        a.bfactor = -1.0
                
                for _, a in u_p2_to_new.items():
                    a.bfactor = 1.0
                
                # Add bonded potentials from p2 that include unique atoms
                processed = set()  # avoid duplicate entries
                for a2 in unique_atoms_j:
                    for pot_map in (bond_map, angle_map, dihed_map, improper_map):
                        for b, rank in pot_map[a2]:
                            # Avoid double-counting
                            if b in processed:
                                continue
                            processed.add(b)
                            
                            # Get atoms involved in the potential
                            if pot_map == bond_map:
                                atoms = (b.atom1, b.atom2)
                                gen_pot = lambda new_atoms: parmed.Bond(*new_atoms, type=b.type)
                                pot_list = new.bonds
                            elif pot_map == angle_map:
                                atoms = (b.atom1, b.atom2, b.atom3)
                                gen_pot = lambda new_atoms: parmed.Angle(*new_atoms, type=b.type)
                                pot_list = new.angles
                            elif pot_map == dihed_map:
                                atoms = (b.atom1, b.atom2, b.atom3, b.atom4)
                                gen_pot = lambda new_atoms: parmed.Dihedral(*new_atoms, type=b.type, improper=b.improper)
                                pot_list = new.dihedrals
                            elif pot_map == improper_map:
                                atoms = (b.atom1, b.atom2, b.atom3, b.atom4)
                                gen_pot = lambda new_atoms: parmed.Improper(*new_atoms, type=b.type)
                                pot_list = new.impropers
                            else:
                                raise Exception("Unknown potential map")
                            
                            # Skip if no atoms in bond are unique atoms
                            if all(a in unique_atoms_j for a in atoms):
                                continue
                            
                            # Find corresponding atoms in new structure
                            new_atoms = []
                            for a in atoms:
                                if a in unique_atoms_j:
                                    new_atoms.append(u_p2_to_new[a])
                                else:
                                    new_atoms.append(new.atoms[p1_to_new_di + p2_to_p1[a].idx])
                            
                            assert all(a in new.atoms for a in new_atoms)
                            
                            # Add potential to new structure
                            pot_list.append(gen_pot(new_atoms))
        
        # Handle base-pair information if MDAnalysis universes are provided
        bp_pairs = None
        if u1 is not None and u2 is not None:
            bp_pairs = []
            
            # Function to find base pairs in a nucleic acid structure
            def find_bp(u):
                # Find segment termini
                seg_ends = [a.position for seg in u.segments
                           for a in [seg.atoms[0], seg.atoms[-1]]]
                seg_ends = np.array(seg_ends)
                
                dists = distance_matrix(seg_ends, seg_ends) + 100 * np.eye(len(seg_ends))
                
                seg_end_map = {}
                for i, d in enumerate(dists):
                    j = np.argmin(d)
                    if i in seg_end_map:
                        assert seg_end_map[i] == j
                    seg_end_map[i] = j
                
                bp = -1 * np.ones(len(u.residues), dtype=int)
                processed = set()
                
                for I, J in seg_end_map.items():
                    if I in processed:
                        continue
                    processed.add(I)
                    processed.add(J)
                    
                    reses1, reses2 = [u.segments[K // 2].residues[::1 - 2 * (K % 2)] for K in (I, J)]
                    for i, (r1, r2) in enumerate(zip(reses1, reses2)):
                        if i == len(reses1) // 2:
                            break
                        bp[r1.resindex] = r2.resindex
                        bp[r2.resindex] = r1.resindex
                
                assert np.all(bp >= 0)
                return bp
            
            # Find base pairs in both structures
            bp1, bp2 = [find_bp(u) for u in (u1, u2)]
            
            # Define atom pairs for base-pairing
            bp_bond_atoms = {
                'DA': [('N1', 'N3'), ('N6', 'O4')],
                'DT': [('N3', 'N1'), ('O4', 'N6')],
                'DC': [('N3', 'N1'), ('O2', 'N2')],
                'DG': [('N1', 'N3'), ('N2', 'O2')],
                'A': [('N1', 'N3'), ('N6', 'O4')],
                'T': [('N3', 'N1'), ('O4', 'N6')],
                'U': [('N3', 'N1'), ('O4', 'N6')],
                'C': [('N3', 'N1'), ('O2', 'N2')],
                'G': [('N1', 'N3'), ('N2', 'O2')]
            }
            
            # Map residue names to keys in bp_bond_atoms
            resname_to_key = {}
            for key in bp_bond_atoms:
                resname_to_key[key] = key
                # Handle modified bases
                if key in ('DA', 'DT', 'DC', 'DG'):
                    for i in range(1, 10):
                        mod_name = f"{key}{i}"
                        resname_to_key[mod_name] = key
            
            # Create base pair bonds
            for ra, (r1b, r2b) in enumerate(zip(bp1, bp2)):
                # Ra pairs with r1b; r2a pairs with r2b
                R1a = p1.residues[ra]
                R2a = p2.residues[ra]
                R1b = p1.residues[r1b] if r1b >= 0 else None
                R2b = p2.residues[r2b] if r2b >= 0 else None
                
                # Add base pair bonds for structure 1
                if R1b is not None and ra < r1b:
                    try:
                        res_key = resname_to_key[R1a.name]
                        for n1, n2 in bp_bond_atoms[res_key]:
                            a = next(x for x in R1a.atoms if x.name == n1)
                            b = next(x for x in R1b.atoms if x.name == n2)
                            
                            # Account for index offsets
                            seg = R1a.segid
                            ai = a.idx + first_new_atom[seg] - first_p1_atom[seg]
                            seg = R1b.segid
                            bi = b.idx + first_new_atom[seg] - first_p1_atom[seg]
                            bp_pairs.append((ai, bi))
                    except KeyError:
                        logger.warning(f"Residue {R1a.name} not found in base pair definitions")
                        continue
                
                # Add base pair bonds for structure 2 (if different)
                if R2b is not None and ra < r2b:
                    # Skip redundant base pairs
                    if R1b is not None and R1a.name == R2a.name and R1b.name == R2b.name:
                        continue
                    
                    try:
                        res_key = resname_to_key[R2a.name]
                        for n1, n2 in bp_bond_atoms[res_key]:
                            a = next(x for x in R2a.atoms if x.name == n1)
                            b = next(x for x in R2b.atoms if x.name == n2)
                            
                            try:
                                ai = u_p2_to_new[a].idx
                                assert u_p2_to_new[a].name == n1
                            except KeyError:
                                seg = R1a.segid
                                a1 = next(x for x in R1a.atoms if x.name == n1)
                                ai = a1.idx + first_new_atom[seg] - first_p1_atom[seg]
                            
                            try:
                                bi = u_p2_to_new[b].idx
                                assert u_p2_to_new[b].name == n2
                            except KeyError:
                                seg = R1b.segid
                                b1 = next(x for x in R1b.atoms if x.name == n2)
                                bi = b1.idx + first_new_atom[seg] - first_p1_atom[seg]
                            
                            bp_pairs.append((ai, bi))
                    except KeyError:
                        logger.warning(f"Residue {R2a.name} not found in base pair definitions")
                        continue
        
        return new, pairs, bp_pairs