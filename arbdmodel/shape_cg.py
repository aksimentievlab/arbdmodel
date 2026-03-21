"""
Shape-based coarse-grained modeling module for protein simulations.

This module provides classes for creating shape-based coarse-grained models
of proteins for use in molecular dynamics simulations.
"""

import numpy as np
from pathlib import Path
from scipy.spatial import KDTree

import parmed
import freesasa

from .logger import logger, devlogger
from .core_objects import Group, ParticleType, PointParticle
from .interactions import AbstractPotential, HarmonicBond
from .model import ArbdModel
from .coords import rotationAboutAxis, read_arbd_coordinates

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

def find_shape_based_sites(fine_positions, N_cg,
                          num_steps=None, learning_schedule=None,
                          weights=None,
                          seed=1234):
    """Find optimal placement of coarse-grained sites based on structure shape.
    
    This function uses a competitive learning algorithm to determine optimal
    placement of coarse-grained sites within a molecular structure, considering
    both geometry and optional weights (like atom masses).
    
    Args:
        fine_positions: Array of atomic positions, shape (n_atoms, 3)
        N_cg: Number of coarse-grained sites to create
        num_steps: Number of learning steps (default: 400*N_cg)
        learning_schedule: Custom schedule for learning parameters (default: None)
        weights: Optional weights for each atom (default: uniform)
        seed: Random seed for reproducibility
        
    Returns:
        Array of positions for coarse-grained sites, shape (N_cg, 3)
    """
    rng = np.random.default_rng(seed=seed)
    
    n_fg = len(fine_positions)

    if weights is None:
        weights = np.ones(n_fg)
    weights = np.array(weights)
    weights = weights / np.sum(weights)    

    if num_steps is None:
        num_steps = 400 * N_cg

    r_fg = fine_positions
    
    # Initialize CG sites by selecting random atoms, weighted by provided weights
    r_cg = r_fg[rng.choice(n_fg, size=N_cg, replace=False, p=weights)]
    if not np.all(np.isfinite(r_cg)):
        raise ValueError("Initial CG positions contain non-finite values")

    # Define learning schedule if not provided
    if learning_schedule is None:
        e0, e1 = 0.3, 0.05  # Initial and final learning rates
        l0, l1 = 0.2 * N_cg, 0.01  # Initial and final neighborhood parameters
        
        epsilon = lambda s: e0 * (e1/e0)**(s/num_steps)
        lambda_ = lambda s: l0 * (l1/l0)**(s/num_steps)

    # Choose random fine particles for each step, weighted by provided weights
    rs = r_fg[rng.choice(n_fg, size=num_steps, p=weights)]
    
    # Competitive learning algorithm
    for step, r in enumerate(rs, 1):
        # Calculate distances to all CG sites
        dr0 = (r[None, :] - r_cg)
        dr0_sq = (dr0**2).sum(axis=-1)

        # Calculate k parameter (number of CG sites closer than each site)
        k = np.array([(dr0_sq < x).sum() for x in dr0_sq])
        if not np.all(np.isfinite(k)):
            raise ValueError("Non-finite values in k array during learning")
        
        # Calculate learning rate for each CG site
        learn_rate = epsilon(step) * np.exp(-k/lambda_(step))

        if not np.all(np.isfinite(learn_rate)):
            raise ValueError("Non-finite values in learning rate during learning")

        # Update CG site positions
        r_cg = r_cg + learn_rate[:, None] * dr0
                
    return r_cg

def get_particle_assignments(fine_sites, coarse_sites, max_distance=20):
    """Assign fine-grained particles to coarse-grained sites.
    
    This function maps each atom to its closest coarse-grained site,
    using a KD-tree for efficient distance calculations.
    
    Args:
        fine_sites: Array of atomic positions, shape (n_atoms, 3)
        coarse_sites: Array of coarse-grained site positions, shape (n_cg, 3)
        max_distance: Maximum distance to consider for assignments
        
    Returns:
        Array of assignments, indicating which CG site each atom belongs to
    """
    t_fg = KDTree(fine_sites)
    t_cg = KDTree(coarse_sites)

    # Build sparse distance matrix between fine and coarse sites
    coo = t_fg.sparse_distance_matrix(t_cg, output_type='coo_matrix', max_distance=max_distance)
    csr = coo.tocsr()

    def _nonzero_row_argmin(csr):
        """Find indices of minimum values in each row of sparse matrix."""
        result = []
        for i in range(csr.shape[0]):
            sl = slice(csr.indptr[i], csr.indptr[i+1])
            if len(csr.data[sl]) == 0:
                raise ValueError(f"Fine particle {i} is too far from all coarse sites to assign")
            else:
                idx = np.argmin(csr.data[sl])
                j = csr.indices[sl][idx]  # column index
                result.append(j)
        return np.array(result, dtype=int)

    assignments = _nonzero_row_argmin(csr)
    assert(len(assignments) == len(fine_sites))
    return assignments


def _get_distances( tree1, tree2, max_distance=20 ):
    """ Returns closest distance of each site in tree1 to sites in tree2 """

    t1 = tree1 # KDTree( sites1 )
    t2 = tree2 # KDTree( sites2 )

    coo = t1.sparse_distance_matrix( t2, output_type='coo_matrix', max_distance=max_distance )

    csr = coo.tocsr()

    result = []
    for i in range(csr.shape[0]):
        sl = slice(csr.indptr[i], csr.indptr[i+1])
        if len(csr.data[sl]) == 0:
            # raise Exception("Some fine particles too far from coarse sites to assign")
            result.append(np.inf)
        else:
            # idx = np.argmin(csr.data[sl])
            result.append( np.min(csr.data[sl]) )
    return np.array(result)


class ShapeCGNonbonded(AbstractPotential):
    """Nonbonded potential for shape-based coarse grained models.
    
    The potential includes electrostatic and Lennard-Jones components.
    """
    def __init__(self, debye_length=10, resolution=0.3, rMin=5, range_=(0, 50), **kwargs):
        AbstractPotential.__init__(self, resolution=resolution, range_=range_, **kwargs)
        self.debye_length = debye_length
        self.maxForce = 50
        self.rMin = rMin

    def potential(self, r, types):
        """Calculate the potential energy at distance r.
        
        Args:
            r: Array of distances
            types: Tuple of (typeA, typeB) particle types
            
        Returns:
            Array of potential energies
        """
        typeA, typeB = types
        
        # Electrostatics
        ld = self.debye_length 
        q1 = typeA.charge
        q2 = typeB.charge
        D = 80  # dielectric of water
        # units "e**2 / (4 * pi * epsilon0 AA)" kcal_mol
        A = 332.06371
        u_elec = (A * q1 * q2 / D) * np.exp(-r / ld) / r 
        
        # Lennard-Jones
        epsilon = 0.5
        if hasattr(typeA, 'nts') or hasattr(typeB, 'nts'):
            epsilon = 1.0
                   
        sigma = 0.5 * (typeA.sigma + typeB.sigma)
        r6 = (sigma / r) ** 6
        r12 = r6 ** 2
        u_lj = 4 * epsilon * (r12 - r6)

        # Modified to be repulsive
        rmin = sigma * 2 ** (1/6)
        try:
            i = np.where(r >= rmin)[0][0]
            u_lj = u_lj - u_lj[i]
            u_lj[i+1:] = 0
        except IndexError:
            # Handle case where r doesn't contain values >= rmin
            u_lj = np.zeros_like(r)
          
        # Combine potentials
        u = u_elec + u_lj
        u[0] = u[1] if len(u) > 1 else 0  # Remove NaN

        # Cap the maximum force
        maxForce = self.maxForce
        if maxForce is not None and len(r) > 1:
            assert(maxForce > 0)
            f = np.diff(u) / np.diff(r)
            f[f > maxForce] = maxForce
            f[f < -maxForce] = -maxForce
            u[0] = 0
            u[1:] = np.cumsum(f * np.diff(r))
        
        u = u - u[-1] if len(u) > 0 else u
        return u

    def filename(self, types=None):
        """Get filename for potential lookup table.
        
        Args:
            types: Tuple of particle types
            
        Returns:
            Filename string
        """
        typeA, typeB = types
        return f"potentials/shapecg_{typeA.name}_{typeB.name}.dat"


class ShapeCGFactory:
    """Factory class for creating coarse-grained protein models.
    
    This class creates reduced-resolution models of proteins based on
    their shape, with customizable resolution and properties.
    """
    def __init__(self, psf, pdb, disordered_residues=None, name='SHCG'):
        """Initialize the factory.
        
        Args:
            psf: Path to PSF file
            pdb: Path to PDB file
            disordered_residues: List of residue indices considered disordered
            name: Name prefix for generated particles
        """
        self.psf = psf
        self.pdb = pdb
        self.name = name

        if disordered_residues is None:
            disordered_residues = []
        
        # Using parmed because MDA sometimes misreads psf attributes
        self._fine = _fine = read_files(psf, pdb)
        self._fine_positions = _fine.coordinates
        self._fine_masses = np.array([a.mass for a in _fine])
        self._fine_charges = np.array([a.charge for a in _fine])
        self._fine_names = np.array([a.name for a in _fine])
        self._fine_resnames = np.array([a.residue.name for a in _fine])
        self._fine_idp = np.array([1 if a.residue.idx in disordered_residues else 0 for a in _fine])
        self._fine_resid = np.array([a.residue.idx for a in _fine])
        self._fine_total_mass = self._fine_masses.sum()
        self._coarse_dict = {}  # dictionary that caches shape-CG model with number of beads as key

    def calc_atom_sasa(self, atom_slice):
        """Calculate solvent accessible surface area for atoms.
        
        Args:
            atom_slice: Slice or indices for atoms to calculate SASA
            
        Returns:
            Tuple of (normalized SASA, atomic radii)
        """
        sl = atom_slice
        try:
            self.atom_radii
        except AttributeError:
            structure = freesasa.Structure()
            structure.addAtoms(
                self._fine_names,
                self._fine_resnames,
                [int(x) for x in self._fine_resid],
                len(self._fine_positions) * 'A',
                *[self._fine_positions[:,i] for i in range(3)]
            )

            res = freesasa.calc(structure)

            self.atom_radii = np.array([structure.radius(i) + 1.4 for i in range(res.nAtoms())])
            self.atom_sasa = np.array([res.atomArea(i) for i in range(res.nAtoms())])
            self.atom_area = 4 * np.pi * self.atom_radii**2

        w = self.atom_sasa[sl] / self.atom_area[sl]
        return w, self.atom_radii[sl]
        
    def get_coarse_protein(self, num_CG_sites=None):
        """Generate a coarse-grained protein model.
        
        Args:
            num_CG_sites: Number of sites to represent the protein (default: based on mass)
            
        Returns:
            Group containing the coarse-grained protein
        """
        if num_CG_sites is None:
            # Default to ~5000 dalton/site
            num_CG_sites = int(np.round(self._fine_total_mass / 5000))
        
        if num_CG_sites not in self._coarse_dict:
            # Find optimal placement of CG sites based on shape
            r_cg = find_shape_based_sites(
                self._fine_positions,
                N_cg=num_CG_sites,
                weights=self._fine_masses
            )

            # Assign atoms to CG sites
            mapping = get_particle_assignments(self._fine_positions, r_cg, max_distance=80)
            self.mapping = mapping
            
            types = []
            parts = []

            # Calculate average residue ID for each CG site for consistent numbering
            cg_fine_resids = [self._fine_resid[mapping == i].mean() for i in range(num_CG_sites)]
            cg_resids = np.argsort(np.argsort(cg_fine_resids)) + 1

            part_by_resid = dict()
            for i, (r, rid) in sorted(enumerate(zip(r_cg, cg_resids)), key=lambda x: x[1][1]):
                sl = (mapping == i)
                rad_gyr0 = np.sqrt(np.mean(((self._fine_positions[sl] - r_cg[i])**2).sum(axis=-1)))
                
                # Calculate effective radius using SASA-weighted approach
                def _calc_sasa_rad():
                    w, radii = self.calc_atom_sasa(sl)
                    w = w / w.sum()  # normalized
                    r_sq = ((self._fine_positions[sl] - r_cg[i])**2).sum(axis=-1) + radii**2
                    alpha = 1
                    return np.sqrt((r_sq**alpha * w).sum())**(1.0/alpha)
                
                _alpha = 1.1
                _eps = 0.8 * 12.78**(1.2 - _alpha)
                rad_gyr = _eps * _calc_sasa_rad()**_alpha
                
                logger.debug(f'CG residue {i}/{num_CG_sites}: rgyr = {rad_gyr0}; rgyr_sasa = {rad_gyr}')

                rad = rad_gyr
                mass = self._fine_masses[sl].sum() 
                charge = self._fine_charges[sl].sum()
                idp = self._fine_idp[sl].sum()

                # Scale diffusivity based on mass
                a = (mass / self._fine_total_mass)**(1/3)
                D = num_CG_sites / a  # heuristic with correct scaling

                name = f'{self.name}_{num_CG_sites:02d}_{i:02d}'
                t = ParticleType(
                    name,
                    mass=mass,
                    sigma=2 * rad / 2**(1/6),
                    charge=charge,
                    idp=idp,
                    diffusivity=D,
                )
                p = PointParticle(t, r, name, resid=rid, residue=rid)
                types.append(t)
                parts.append(p)
                part_by_resid[rid] = p
                
            g = Group(name=self.name, segname=self.name, children=parts)

            # Add all-to-all elastic network for structured region
            num_idp = sum([p.idp > 0.5 for p in parts])        
            for i, p1 in enumerate(parts[:-1]):
                if p1.idp > 0.5:
                    continue
                for j, p2 in enumerate(parts[i+1:], i+1):
                    if p2.idp > 0.5:
                        continue
                    r0 = np.linalg.norm((p2.position - p1.position))
                    g.add_bond(
                        i=p1, 
                        j=p2, 
                        bond=HarmonicBond(k=50/(num_CG_sites-num_idp)**2, r0=r0), 
                        exclude=True
                    )

            # Add bonds to nearest beads for idp regions
            idp_parts = [p for p in parts if p.idp > 0.5]
            _idp_bonds = set()
            for p1 in idp_parts:
                try:
                    p2 = part_by_resid[p1.resid+1]
                    _idp_bonds.add((p1, p2))
                except KeyError:
                    pass
                try:
                    p2 = part_by_resid[p1.resid-1]
                    _idp_bonds.add((p2, p1))
                except KeyError:
                    pass
                    
            for p1, p2 in _idp_bonds:
                r0 = np.linalg.norm((p2.position - p1.position))
                g.add_bond(i=p1, j=p2, bond=HarmonicBond(k=1, r0=r0), exclude=True)
                
            self._coarse_dict[num_CG_sites] = (g, types)
            
        return self._coarse_dict[num_CG_sites][0]

    def get_coarse_types(self, num_CG_sites=None):
        """Get particle types used in the coarse-grained model.
        
        Args:
            num_CG_sites: Number of sites to represent the protein
            
        Returns:
            List of ParticleType objects
        """
        if num_CG_sites is None:
            # Default to ~5000 dalton/site
            num_CG_sites = int(np.round(self._fine_total_mass / 5000))

        if num_CG_sites not in self._coarse_dict:
            self.get_coarse_protein(num_CG_sites)
            
        return self._coarse_dict[num_CG_sites][1]

    def generate_protein(self, position, orientation=None, index=0, num_CG_sites=None):
        """Generate a new protein instance at the specified position and orientation.
        
        Args:
            position: 3D position for the protein
            orientation: Optional orientation matrix
            index: Unique index for this protein instance
            num_CG_sites: Number of CG sites to use
            
        Returns:
            Group containing the positioned protein
        """
        new = self.get_coarse_protein(num_CG_sites).duplicate()
        if orientation is not None:
            new.orientation = orientation
        new.position = position

        new.segname = f'{new.name}{index:03d}'
        new.name = new.segname
        return new


class ShapeCGModel(ArbdModel):
    """Model class for shape-based coarse-grained protein simulations.
    
    This class handles creating and simulating systems with multiple
    shape-based coarse-grained proteins.
    """
    def __init__(self, protein_factories=None, box_size=None, **kwargs):
        """Initialize the model.
        
        Args:
            protein_factories: Dictionary mapping protein names to ShapeCGFactory objects
            box_size: Size of simulation box (default: None for automatic sizing)
            **kwargs: Additional arguments for ArbdModel
        """
        dimensions = [box_size, box_size, box_size] if box_size else None
        ArbdModel.__init__(self, [], dimensions=dimensions, **kwargs)
        
        self.protein_factories = protein_factories or {}
        self.protein_types_cache = {}  # Cache for protein types
        
    def concentration_to_debye_length(self, c, temperature=295):
        """Convert salt concentration to Debye length.
        
        Args:
            c: Salt concentration in mM
            temperature: Temperature in K
            
        Returns:
            Debye length in Angstroms
        """
        # sqrt(80 * epsilon0 * 295 K / ((150 mM) * e**2/particle))
        return 11.154259 * np.sqrt((150/c) * (295/temperature))
        
    def add_protein_nb_interactions(self, prot_types, nt_sigma=10, debye_length=None):
        """Add nonbonded interactions between protein particles.
        
        Args:
            prot_types: List of protein particle types
            nt_sigma: Sigma parameter for nucleic acid interactions
            debye_length: Debye length for electrostatics (default: calculated from 150 mM)
        """
        if debye_length is None:
            debye_length = self.concentration_to_debye_length(150)

        prot_nb = ShapeCGNonbonded(debye_length=debye_length)
        old_interactions = self.nonbonded_interactions
        self.nonbonded_interactions = []

        # Add protein-protein interactions
        for i, prot_t1 in enumerate(prot_types):
            for j, prot_t2 in enumerate(prot_types[i:], i):
                self.add_nonbonded_interaction(prot_nb, typeA=prot_t1, typeB=prot_t2)
            
            # Add protein-nucleic acid interactions if present
            try:
                for t in self.nucleic_types:
                    if t.name[0] == 'O':
                        continue
                    t.charge = -1 * t.nts
                    t.sigma = nt_sigma
                    self.add_nonbonded_interaction(prot_nb, typeA=t, typeB=prot_t1)
            except AttributeError:
                pass
                
        # Restore previous interactions
        self.nonbonded_interactions = self.nonbonded_interactions + old_interactions
        
    def get_protein_types(self, protein_counts, num_CG_sites=None):
        """Get all protein types for the given protein counts.
        
        Args:
            protein_counts: Dictionary mapping protein names to counts
            num_CG_sites: Number of CG sites per protein
            
        Returns:
            List of all protein particle types
        """
        cache_key = (tuple(sorted(protein_counts.items())), num_CG_sites)
        if cache_key in self.protein_types_cache:
            return self.protein_types_cache[cache_key]
            
        prot_types = []
        for k, count in protein_counts.items():
            if count == 0 or k not in self.protein_factories:
                continue
            prot_types.extend(self.protein_factories[k].get_coarse_types(num_CG_sites))
            
        self.protein_types_cache[cache_key] = prot_types
        return prot_types
        
    def add_proteins(self, protein_counts, protein_positions, num_CG_sites=None, nt_sigma=10):
        """Add proteins to the simulation model.
        
        Args:
            protein_counts: Dictionary mapping protein names to counts
            protein_positions: List of positions for all proteins
            num_CG_sites: Number of CG sites per protein
            nt_sigma: Sigma parameter for nucleic acid interactions
            
        Returns:
            Group containing all added proteins
        """
        all_pro = []
        start = 0
        
        # Create proteins and add them to the model
        for k, count in protein_counts.items():
            if count == 0 or k not in self.protein_factories:
                continue
                
            pro_parts = [
                self.protein_factories[k].generate_protein(
                    protein_positions[start + i], 
                    index=i,
                    num_CG_sites=num_CG_sites
                ) for i in range(count)
            ]
            
            g = Group(name=f'all_{k}', children=pro_parts)
            all_pro.append(g)
            start += count

        # Create a group containing all proteins
        protein_group = Group(name='cytosol', children=all_pro)
        self.add(protein_group)
        
        # Add nonbonded interactions
        prot_types = self.get_protein_types(protein_counts, num_CG_sites)
        self.add_protein_nb_interactions(prot_types, nt_sigma=nt_sigma)
        
        return protein_group
        
    def generate_random_protein_positions(self, protein_counts, radius, seed=None):
        """Generate random positions for proteins within a spherical volume.
        
        Args:
            protein_counts: Dictionary mapping protein names to counts
            radius: Radius of the sphere
            seed: Random seed (default: None)
            
        Returns:
            List of 3D positions
        """
        if seed is not None:
            np.random.seed(seed)
            
        total_proteins = sum(protein_counts.values())
        
        # Generate random positions on a sphere with random radii
        _x, _y, _z = np.eye(3)
        
        angles = np.random.random((total_proteins, 2))
        angles[:, 0] = angles[:, 0] * 360  # Azimuthal angle
        angles[:, 1] = np.arccos(1 - 2 * angles[:, 1]) * 180/np.pi  # Polar angle

        # Create rotation matrices
        pro_rots = [
            rotationAboutAxis(_z, a0).dot(rotationAboutAxis(_x, a1))
            for a0, a1 in angles
        ]
        
        # Generate radii with cubic distribution for uniform volume sampling
        radii = np.random.random((total_proteins, 1))**(1/3) * radius
        
        # Final positions
        pro_centers = [R[:, 2] * r for R, r in zip(pro_rots, radii)]
        
        return pro_centers
        
    def setup_and_run(self, protein_counts, radius=200, num_CG_sites=None, 
                     salt_concentration=150, gpu=0, dry_run=True, 
                     num_steps=1e8, output_period=1e4, timestep=200e-6,
                     positions=None, restart_file=None):
        """Set up and run a simulation with the specified parameters.
        
        Args:
            protein_counts: Dictionary mapping protein names to counts
            radius: Confinement radius
            num_CG_sites: Number of CG sites per protein
            salt_concentration: Salt concentration in mM
            gpu: GPU index to use
            dry_run: If True, don't actually run the simulation
            num_steps: Number of simulation steps
            output_period: Steps between outputs
            timestep: Simulation timestep
            positions: Optional pre-defined positions for proteins
            restart_file: Optional restart file path
            
        Returns:
            Final simulation state
        """
        # Set up system parameters
        debye_length = self.concentration_to_debye_length(salt_concentration)
        
        # Get number of proteins
        total_proteins = sum(protein_counts.values())
        logger.info(f"Setting up system with {total_proteins} proteins")
        
        # Generate positions if not provided
        if positions is None:
            positions = self.generate_random_protein_positions(protein_counts, radius)
        
        # Set up simulation name and directory
        name = f'CG{num_CG_sites}_r{radius}_salt{salt_concentration}'
        directory = f'run_{num_CG_sites}'
        
        # Add proteins to model
        protein_group = self.add_proteins(
            protein_counts,
            positions,
            num_CG_sites=num_CG_sites
        )
        
        # Add confinement to protein particles
        self.add_confinement(protein_group, radius)
        
        # Run the simulation
        self.simulate(
            name,
            directory=directory,
            num_steps=num_steps,
            output_period=output_period,
            timestep=timestep,
            gpu=gpu,
            restart_file=restart_file,
            dry_run=dry_run
        )
        
        return self
        
    def add_confinement(self, protein_group, radius):
        """Add spherical confinement to all protein particles.
        
        Args:
            protein_group: Group containing proteins
            radius: Confinement radius
        """
        # Make sure the confinement file exists
        confinement_file = f'../confine-{radius}.dx'
        if not Path(confinement_file).exists():
            # Create confinement file
            from .grid import write_confine_dx
            write_confine_dx(radius=radius)
        
        # Add confinement to all protein particles
        for t in set([p.type_ for p in protein_group]):
            t.grid = [(confinement_file, 1)]
            
        # Add confinement to all nonbonded scheme particle types
        for s, t1, t2 in self.nonbonded_interactions:
            for t in (t1, t2):
                t.grid = [(confinement_file, 1)]
                
    def run_minimization(self, protein_counts, radius=200, gpu=0, 
                        num_steps=1e5, output_period=1e4, timestep=200e-6,
                        directory='run_1'):
        """Run a minimization simulation with minimal CG sites.
        
        Args:
            protein_counts: Dictionary mapping protein names to counts
            radius: Confinement radius
            gpu: GPU index to use
            num_steps: Number of minimization steps
            output_period: Steps between outputs
            timestep: Simulation timestep
            directory: Output directory
            
        Returns:
            Path to restart file
        """
        # Generate random positions
        positions = self.generate_random_protein_positions(protein_counts, radius)
        
        # Create model with only 1 CG site per protein for quick minimization
        protein_group = self.add_proteins(
            protein_counts,
            positions,
            num_CG_sites=1
        )
            
        # Add confinement with stronger parameters for minimization
        for t in set([p.type_ for p in protein_group]):
            t.modified = True
            t.sigma = 2 * t.sigma
            t.grid = [(f'../confine-{radius}.dx', 1)]

        for s, t1, t2 in self.nonbonded_interactions:
            for t in (t1, t2):
                try:
                    t.modified
                except AttributeError:
                    t.modified = True
                    t.sigma = 2 * t.sigma
                    t.grid = [(f'../confine-{radius}.dx', 1)]

        # Run minimization
        minimization_name = f'min_{radius}'
        self.simulate(
            minimization_name,
            directory=directory,
            num_steps=num_steps,
            output_period=output_period,
            timestep=timestep,
            gpu=gpu,
            dry_run=False
        )
        
        return f'{directory}/output/{minimization_name}.restart'
        
    def run_from_minimized(self, protein_counts, radius=200, num_CG_sites=None,
                          salt_concentration=150, gpu=0, dry_run=True,
                          num_steps=1e8, output_period=1e4, timestep=200e-6,
                          minimization_file=None):
        """Run a full simulation starting from a minimized configuration.
        
        Args:
            protein_counts: Dictionary mapping protein names to counts
            radius: Confinement radius
            num_CG_sites: Number of CG sites per protein
            salt_concentration: Salt concentration in mM
            gpu: GPU index to use
            dry_run: If True, don't actually run the simulation
            num_steps: Number of simulation steps
            output_period: Steps between outputs
            timestep: Simulation timestep
            minimization_file: Path to minimization restart file
            
        Returns:
            Final simulation state
        """
        # Check for restart file or run minimization
        if minimization_file is None or not Path(minimization_file).exists():
            logger.info("No minimization file found, running minimization first")
            minimization_file = self.run_minimization(protein_counts, radius, gpu)
        
        # Clear the model and restart
        self.clear_all()
        
        # Read coordinates from restart file
        new_protein_coords = read_arbd_coordinates(minimization_file)[len(self):]
        
        # Set up simulation name and directory
        name = f'CG{num_CG_sites}_r{radius}_salt{salt_concentration}'
        directory = f'run_{num_CG_sites}'
        
        # Add proteins with proper CG resolution
        protein_group = self.add_proteins(
            protein_counts,
            new_protein_coords,
            num_CG_sites=num_CG_sites
        )
        
        # Add confinement to protein particles
        self.add_confinement(protein_group, radius)
        
        # Run the simulation
        self.simulate(
            name,
            directory=directory,
            num_steps=num_steps,
            output_period=output_period,
            timestep=timestep,
            gpu=gpu,
            dry_run=dry_run
        )
        
        return self
        
    @classmethod
    def from_protein_list(cls, pdb_list, pdb_paths=None, **kwargs):
        """Create a model from a list of protein PDB IDs.
        
        Args:
            pdb_list: List of protein PDB IDs
            pdb_paths: Optional dictionary mapping PDB IDs to PSF/PDB file paths
            **kwargs: Additional arguments for the ShapeCGModel constructor
            
        Returns:
            ShapeCGModel instance
        """
        # Create short names for proteins
        pdb_name = {k: k[0] + k[-2:] for k in pdb_list}
        
        # Create factory objects for each protein
        protein_factories = {}
        for k in pdb_list:
            if pdb_paths and k in pdb_paths:
                psf = pdb_paths[k]['psf']
                pdb = pdb_paths[k]['pdb']
            else:
                psf = f'{k}_autopsf.psf'
                pdb = f'{k}_autopsf.pdb'
                
            protein_factories[k] = ShapeCGFactory(psf, pdb, name=pdb_name[k])
            
        return cls(protein_factories=protein_factories, **kwargs)
        
    def calculate_total_mass(self, protein_counts):
        """Calculate total mass of proteins.
        
        Args:
            protein_counts: Dictionary mapping protein names to counts
            
        Returns:
            Total mass in daltons
        """
        return sum([
            self.protein_factories[k]._fine_total_mass * count
            for k, count in protein_counts.items()
            if k in self.protein_factories
        ])