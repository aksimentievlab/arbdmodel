#!/usr/bin/env python3
import os
from pathlib import Path
from scipy.cluster import vq
import numpy as np
from .model import ArbdModel
from .config import DefaultSimConf
from .logger import logger
from .coords import Generate_coordinates, Generate_spanning_vectors
from .core_objects import RigidBody, Group
from .rb_from_pdb import DiffusiveRigidBodyType, StaticObject

class RBContactModel(ArbdModel):
    """Model class for structure-based rigid body simulations"""
    
    def __init__(self, cell_vectors=None, cell_origin=None,
                 dimensions=None, buffer_factor=1.2, configuration=None, use_boundary=False,
                 num_heavy_cluster=3, charmm_params_dir=None, gaussian_width=None,
                 pot_resolution=1, den_resolution=2,
                 boundary_params=None, work_dir=None, **kwargs):
        """Initialize structure model Former SimpleARBD.
        Args:
            diffusible_objects: List of RBContact instances for diffusible objects
            static_objects: List of RBContact instances for static objects
            cell_vectors: List of 3 cell basis vectors defining the simulation box
            cell_origin: Cell origin coordinates
            dimensions: Explicit dimensions for the simulation box (overrides cell_vectors)
            buffer_factor: Factor to scale dimensions derived from cell vectors
            configuration: Configuration object, defaults to DefaultSimConf
            use_boundary: Whether to create a boundary potential
            boundary_params: Optional parameters for boundary potential (well_depth, resolution, etc.)
            work_dir: Directory for pooled ``clustered.txt`` and default layout for child processors.
            **kwargs: Additional arguments passed to ArbdModel
        """
        
        self.simconf = configuration or DefaultSimConf()
        self.diffusible_objects = []
        self.static_objects = []
        self.boundary_potential = None
        self.initial_positions = {}  # Store initial positions for each type
        self.num_heavy_cluster = num_heavy_cluster
        self.charmm_params_dir = charmm_params_dir
        self.gaussian_width = gaussian_width if gaussian_width is not None else 2.5
        self.pot_resolution = pot_resolution
        self.den_resolution = den_resolution
        self._diffusible_rb_types = []
        self.shared_cluster_file = None
        self.work_dir = Path(work_dir) if work_dir is not None else Path.cwd()

        super().__init__(
            children=[], 
            cell_vectors=cell_vectors, 
            cell_origin=cell_origin,
            dimensions=dimensions,
            buffer_factor=buffer_factor,
            configuration=self.simconf,
            **kwargs
        )
        
        # Process boundary if requested
        if use_boundary:
            logger.info("Creating boundary potential")
            # Create boundary potential
            from .interactions import BoundaryPotential
            
            # Use provided boundary parameters or defaults
            bp_params = boundary_params or {}
            
            boundary = BoundaryPotential(
                cell_vectors=cell_vectors,
                cell_origin=cell_origin,
                well_depth=bp_params.get('well_depth', 1.0),
                resolution=bp_params.get('resolution', 2.0),
                blur=bp_params.get('blur', 5.0)
            )
            
            # Generate the boundary file and store it
            boundary_file = boundary.write_file(bp_params.get('output_file', 'boundary.dx'))
            self.boundary_potential = boundary_file
            #self.add_nonbonded_interaction(self.boundary_potential)

    def add(self, obj):
        """Register contact-model objects or delegate to ``ArbdModel.add``.

        ``DiffusiveRigidBodyType`` instances are collected for pooled LJ clustering
        in :meth:`build_vdw_maps`. ``StaticObject`` instances are processed after
        ``build_vdw_maps`` (cluster file and resolutions are injected from the model).
        ``RigidBody``, ``Group``, and other types use the standard ARBD model logic.
        """
        if isinstance(obj, DiffusiveRigidBodyType):
            self._diffusible_rb_types.append(obj)
            self.shared_cluster_file = None
            return obj
        if isinstance(obj, StaticObject):
            if self.shared_cluster_file is None:
                raise RuntimeError(
                    "Call model.build_vdw_maps() before model.add(StaticObject)."
                )
            obj.cluster_file = Path(self.shared_cluster_file)
            obj.pot_resolution = self.pot_resolution
            obj.den_resolution = self.den_resolution
            if obj.charmm_params_dir is None and self.charmm_params_dir is not None:
                obj.charmm_params_dir = Path(self.charmm_params_dir).expanduser().resolve()
            obj.process()
            obj.smooth_grids(gaussian_width=self.gaussian_width)
            self.static_objects.append(obj)
            return obj
        ret = super().add(obj)
        self._track_diffusible_rigid_bodies(obj)
        return ret

    def _track_diffusible_rigid_bodies(self, obj):
        """Record rigid bodies whose type is DiffusiveRigidBodyType (for I/O helpers)."""
        if isinstance(obj, RigidBody):
            if isinstance(obj.type_, DiffusiveRigidBodyType) and obj not in self.diffusible_objects:
                self.diffusible_objects.append(obj)
        elif isinstance(obj, Group):
            for child in obj.children:
                self._track_diffusible_rigid_bodies(child)

    def add_diffusible_object(self, structure_path,  copies=1, positions=None, 
                         orientations=None, name=None, initial_region=None, random_seed=None):
        """Add a diffusible (mobile) rigid body type to the model.
        
        Args:
            structure_path: Path to structure files (.psf/.pdb) to create a RBContactType
            copies: Number of copies to add
            positions: Optional list of positions for each copy
            orientations: Optional list of orientations for each copy
            name: Optional base name for the rigid bodies
            initial_region: Optional dict with vectors defining initial region
            random_seed: Optional seed for random number generator
        
        Returns:
            List of created RigidBody instances
        """
        # Process the structure to create a RigidBodyType if structure_path provided

            # Create a RBContactType from the structure files
        if name is None:
            name = Path(structure_path).stem
        logger.info(f"Processing structure files for {name} from {structure_path}")
            
            # Create the RigidBodyType using rbs/name directory
        rb_type = DiffusiveRigidBodyType(
            name=name,
            structure_path=structure_path,
            simconf=self.simconf,
            work_dir=self.work_dir,
            num_heavy_cluster=self.num_heavy_cluster,
            charmm_params_dir=self.charmm_params_dir,
        )
        self.add(rb_type)

        logger.info(f"Created RBType for {name}")
            
        # Create base name for rigid bodies of this type
        base_name = name or rb_type.name
        created_bodies = []
        
        # Generate positions if not provided
        if positions is None and copies > 0:
            # Set up default initial region if not provided
            if initial_region is None:
                max_dim = max(self.dimensions)
                initial_region = {
                    'bv1': [max_dim*0.8, 0, 0],
                    'bv2': [0, max_dim*0.8, 0],
                    'bv3': [0, 0, max_dim*0.8],
                    'origin': [0, 0, 0]
                }
                
            # Get dimensions for this rigid body type
            dimensions = [100, 100, 100]  # Default
            
            # Try to get actual dimensions from structure
            if hasattr(rb_type, 'aligned_pdb') and rb_type.aligned_pdb:
                try:
                    # Try to read dimensions from dimension.dat file in the same directory
                    dim_file = rb_type.aligned_pdb.parent / f"{rb_type.aligned_pdb.stem}.dimension.dat"
                    if dim_file.exists():
                        with open(dim_file) as f:
                            dimensions = [float(line.strip()) for line in f.readlines()]
                        logger.info(f"Using dimensions from {dim_file}: {dimensions}")
                except Exception as e:
                    logger.warning(f"Could not read dimensions from file: {e}")
            
            # Generate spanning vectors
            bv1, bv2, bv3, n1, n2, n3 = Generate_spanning_vectors(
                initial_region['bv1'], 
                initial_region['bv2'], 
                initial_region['bv3'],
                dimensions
            )
            
            # Set random seed if provided
            if random_seed is not None:
                prev_state = np.random.get_state()
                np.random.seed(random_seed)
            
            # Generate coordinates
            generated_coords = Generate_coordinates(
                bv1, bv2, bv3, n1, n2, n3, copies,
                initial_region['origin'], random_seed or 0
            )
            
            # Restore random state if we changed it
            if random_seed is not None:
                np.random.set_state(prev_state)
            
            # Store positions for later use
            self.initial_positions[rb_type.name] = generated_coords
            positions = generated_coords
            
            logger.info(f"Generated {copies} initial positions for {rb_type.name}")
        
        # Set up default orientation if none provided
        default_orientation = np.eye(3)
        
        # Add requested number of copies
        for i in range(copies):
            # Create position and orientation for this copy
            if positions is not None and i < len(positions):
                position = positions[i]
            else:
                # Generate random position within the model bounds
                position = np.random.uniform(-0.4, 0.4, 3) * self.dimensions
                
            if orientations is not None and i < len(orientations):
                orientation = orientations[i]
            else:
                orientation = default_orientation
            
            # Create a new rigid body
            rb_name = f"{base_name}_{i+1}" if copies > 1 else base_name
            rb = RigidBody(
                type_=rb_type,
                position=position,
                orientation=orientation,
                name=rb_name)

            self.add(rb)
            created_bodies.append(rb)
            
        return created_bodies

    def _run_pooled_clustering(self, all_records, n_heavy):
        """Run pooled k-means over all LJ records and write a shared cluster file."""
        heavy_records = [r for r in all_records if r["group"] == "heavy"]
        hyd_records = [r for r in all_records if r["group"] == "hydrogen"]
        if not heavy_records and not hyd_records:
            raise ValueError("No LJ type data was generated for pooled clustering.")

        heavy_points = np.array([[r["radius"], r["epsilon"]] for r in heavy_records], dtype=float) if heavy_records else np.empty((0, 2))
        hyd_points = np.array([[r["radius"], r["epsilon"]] for r in hyd_records], dtype=float) if hyd_records else np.empty((0, 2))

        def _expand_by_count(records):
            expanded = []
            for rec in records:
                count = int(rec.get("count", 1))
                if count <= 0:
                    continue
                expanded.append(np.repeat([[rec["radius"], rec["epsilon"]]], count, axis=0))
            if not expanded:
                return np.empty((0, 2))
            return np.vstack(expanded)

        def _cluster_points(points, weighted_points, n_clusters):
            if points.shape[0] == 0 or n_clusters == 0:
                return np.empty((0,), dtype=int), np.empty((0, 2))
            if points.shape[0] == 1:
                return np.array([0], dtype=int), points.copy()

            data = weighted_points if weighted_points.shape[0] > 0 else points
            means = np.mean(data, axis=0)
            scales = np.std(data, axis=0)
            scales[scales == 0] = 1.0
            normalized = (data - means) / scales

            cluster_count = min(n_clusters, data.shape[0])
            codebook, _ = vq.kmeans(normalized, cluster_count)
            codebook = codebook * scales + means
            assignments, _ = vq.vq(points, codebook)
            return assignments, codebook

        np.random.seed(seed=42)
        heavy_weighted = _expand_by_count(heavy_records)
        hyd_weighted = _expand_by_count(hyd_records)
        heavy_clusters = min(n_heavy, heavy_points.shape[0]) if heavy_points.shape[0] > 0 else 0
        assignments_heavy, codebook_heavy = _cluster_points(heavy_points, heavy_weighted, heavy_clusters)
        assignments_hyd, codebook_hyd = _cluster_points(hyd_points, hyd_weighted, 1 if hyd_points.shape[0] > 0 else 0)

        if assignments_hyd.size > 0:
            assignments_total = np.concatenate((assignments_heavy, assignments_hyd + heavy_clusters), axis=None)
            codebook_total = np.concatenate((codebook_heavy, codebook_hyd), axis=0)
        else:
            assignments_total = assignments_heavy
            codebook_total = codebook_heavy

        ordered_records = heavy_records + hyd_records
        type_array = {i: "" for i in range(len(set(assignments_total)))}
        for assign, record in zip(assignments_total, ordered_records):
            type_array[assign] = f"{type_array[assign]} {record['types']}"

        cluster_file = self.work_dir / "clustered.txt"
        with open(cluster_file, "w") as f:
            for i, (radius, epsilon) in enumerate(codebook_total):
                f.write(f"{radius} {epsilon}{type_array[i]}\n")
        return cluster_file

    def build_vdw_maps(self):
        """Pool LJ records across all diffusible processors and build shared VDW maps."""
        if not self._diffusible_rb_types:
            logger.info("No diffusible rigid-body types to finalize clustering for.")
            return None

        all_records = []
        for rb_type in self._diffusible_rb_types:
            all_records.extend(rb_type.processor.lj_type_records)

        cluster_file = self._run_pooled_clustering(all_records, self.num_heavy_cluster)
        self.shared_cluster_file = cluster_file

        for rb_type in self._diffusible_rb_types:
            rb_type.finalize_grids(
                cluster_file,
                gaussian_width=self.gaussian_width,
                potResolution=self.pot_resolution,
                denResolution=self.den_resolution,
            )

        logger.info(f"Pooled LJ clustering complete. Shared cluster file: {cluster_file}")
        return cluster_file

    def finalize_diffusible_vdw_clustering(self):
        """Backward-compatible alias for build_vdw_maps()."""
        return self.build_vdw_maps()

    def wire_static_interactions(self, wire_particles=True):
        """Inject static object grids into diffusive rigid-body types and, optionally, particle types.

        Call this after all static objects have been added via ``model.add(static_obj)``.
        ``generate_all_structures()`` calls this automatically.

        For each diffusive ``RigidBodyType``:
          • its ``pmf_grids`` list receives every ``(keyword, path, scale)`` tuple from each
            ``StaticObject.potential_grids`` — these become ``gridFile`` entries in the ``.bd``
            file so ARBD applies the static field as a background PMF to the rigid body.

        For each free particle type already registered in the model (``wire_particles=True``):
          • ``grid_potentials`` is extended with ``(path, scale)`` from every static potential grid
            so particles also feel the static background field.
        """
        if not self.static_objects:
            return

        # ── diffusive rigid-body types ──────────────────────────────────────
        for rb_type in self._diffusible_rb_types:
            if not isinstance(rb_type.pmf_grids, list):
                rb_type.pmf_grids = list(rb_type.pmf_grids)
            existing = {(k, str(g)) for k, g, *_ in rb_type.pmf_grids}
            for static in self.static_objects:
                for item in static.potential_grids:
                    kw, path = item[0], item[1]
                    scale = item[2] if len(item) == 3 else 1.0
                    if (kw, str(path)) not in existing:
                        rb_type.pmf_grids.append((kw, str(path), scale))
                        existing.add((kw, str(path)))
            logger.info(
                f"Wired {len(self.static_objects)} static object(s) as PMF grids "
                f"into diffusive RB type '{rb_type.name}'"
            )

        # ── free particle types ─────────────────────────────────────────────
        if not wire_particles:
            return
        try:
            pt_counts = list(self.getParticleTypesAndCounts())
        except Exception:
            return
        for pt, (num, num_rb) in pt_counts:
            if num == 0:
                continue
            if not hasattr(pt, "grid_potentials"):
                pt.grid_potentials = []
            elif not isinstance(pt.grid_potentials, list):
                pt.grid_potentials = list(pt.grid_potentials)
            existing_pt = {str(g) for g, *_ in pt.grid_potentials}
            for static in self.static_objects:
                for item in static.potential_grids:
                    path = str(item[1])
                    scale = item[2] if len(item) == 3 else 1.0
                    if path not in existing_pt:
                        pt.grid_potentials.append((path, scale, "dirichlet"))
                        existing_pt.add(path)

    def generate_all_structures(self):
        """Finalize model structure generation and wire static-field interactions.

        Ensures VDW maps exist (calls :meth:`build_vdw_maps` if needed), then wires
        each static object's potential grids into the diffusive rigid-body types as
        PMF background fields and into free particle types as grid potentials.
        """
        if self.shared_cluster_file is None:
            self.build_vdw_maps()
        self.wire_static_interactions()
        logger.info("All model structures are generated and ready for simulation files.")
        return self.shared_cluster_file

    def add_static_object(self, structure_path, is_gigantic=False, threshold=300, work_dir=None):
        """
        Adds a static object to the simulation.
        
        This method processes the specified static structure and keeps its
        electrostatic and potential/charge grids for static-field usage.
        
        Parameters
        ----------
        structure_path : str
            Path to the structure file of the static object.
        is_gigantic : bool, default=False
            Flag indicating whether the structure is exceptionally large.
        threshold : int, default=300
            Size threshold for processing large structures.
        
        Returns
        -------
        StaticObject
            The created and added static object.
        """
        name = Path(structure_path).stem
        static_dir = Path(work_dir) if work_dir else self.work_dir / "static" / name
        os.makedirs(static_dir, exist_ok=True)

        obj = StaticObject(
            structure_path=structure_path,
            name=name,
            simconf=self.simconf,
            work_dir=static_dir,
            is_gigantic=is_gigantic,
            threshold=threshold,
            pot_resolution=self.pot_resolution,
            den_resolution=self.den_resolution,
            charmm_params_dir=self.charmm_params_dir,
        )
        """
        # Add potential grids to model
        for grid_type, grid_file, scale in obj.potential_grids:
            self.add_nonbonded_interaction(grid_type, grid_file, scale)
            
        # Add charge grids to model
        for grid_type, grid_file in obj.charge_grids:
            self.add_nonbonded_interaction(grid_type, grid_file)
            
        # Add electrostatic grid if available
        if obj.elec_grid:
            self.add_nonbonded_interaction("elec", obj.elec_grid, 0.59616195)
        """
        return self.add(obj)
