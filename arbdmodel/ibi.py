from abc import ABCMeta

from scipy.ndimage import label, distance_transform_edt
from scipy.spatial import KDTree
from scipy.signal import savgol_filter as savgol
from scipy.interpolate import RBFInterpolator
from .util import savgol_filter_nd as savgol_nd

import numpy as np
from pathlib import Path
from . import ArbdModel
from . import logger
from .interactions import AbstractPotential

from gridData import Grid

from tqdm import tqdm,trange
from tqdm.contrib.logging import logging_redirect_tqdm


""" This file includes various routines to simplify running
simulations with Iterative Boltmann Inversion. """


class DegreeOfFreedomGroup():
    """ Streamline calculation of IBI degrees of freedom by consolidating functions """
    ...

class DegreeOfFreedom():
    """ Base class for representing a degree of freedom in the system.

    Concrete examples include the distance, angle, or dihedral angle,
 between nearby "bonded" interaction sites, or the distances between
 pairs of particle types (with exclusions!).

    Derived classes should calculate a value for the DoF from particle
  coordinates. """


    def __init__(self,*particles):
        self.particles = particles
        # self.ids = [p.idx for p in particles]
        self.sels = dict()
        self.current_key = None

    def _particle_to_sel(particle, universe):

        """ Recursive function to convert particle or group_site to
        selection or list of selections """

        if isinstance(particle, ArbdModel._GroupSite):
            return [DegreeOfFreedom._particle_to_sel(p,universe)
                    for p in particle.particles]
        else:
            return universe.select_atoms( "index {}".format(particle.idx) )

    def _sel_list_to_positions(sel_list):

        """ Recursive function to convert list of selctions into positions """

        positions = []
        for sel in sel_list:
            if isinstance(sel, list):
                positions.append( DegreeOfFreedom._sel_list_to_positions(sel).mean(axis=0) )
            else:
                positions.append( sel.positions.mean(axis=0) )
        return positions

    def wrap_vector(self, v):
        """
        Adjusts a vector to account for periodic boundary conditions.

        This method modifies the input vector `v` such that its components
        are wrapped within the periodic boundaries of the simulation box.
        It assumes that the simulation box has orthogonal dimensions (all angles are 90 degrees).

        Parameters:
            v (numpy.ndarray): A NumPy array representing the vector(s) to be wrapped.
                               The array can have any shape, but the last dimension
                               should correspond to the 3 spatial coordinates (x, y, z).

        Returns:
            numpy.ndarray: The wrapped vector(s) with the same shape as the input `v`.

        Raises:
            AssertionError: If the simulation box does not have orthogonal dimensions
                            (i.e., any of the angles are not 90 degrees).
        """
        box = self.sels[self.current_key][0].universe.dimensions
        assert(np.all( np.array(box[3:]) == 90 ))

        for i in range(3):
            sl = (v[...,i] > 0.5*box[i])
            v[sl,i] = v[sl,i] - box[i]

            sl = (v[...,i] < -0.5*box[i])
            v[sl,i] = v[sl,i] + box[i]
        return v

    def get_values(self, universe):
        """
        Retrieve computed values based on the current state of the universe.

        This method calculates and returns a list of values derived from the 
        positions of particles in the given universe. It uses a caching mechanism 
        to store selections for previously processed universes, identified by 
        their hash keys.

        Args:
            universe: An object representing the current state of the universe. 
                      It must implement a `__hash__` method to uniquely identify 
                      its state.

        Returns:
            list: A list containing the computed value(s) based on the positions 
                  of the particles in the universe.

        Raises:
            AssertionError: If the result contains NaN values or if the result 
                            list is empty.
        """
        key = universe.__hash__()
        if key not in self.sels:
            self.sels[key] = [DegreeOfFreedom._particle_to_sel(p,universe)
                              for p in self.particles]
        positions = DegreeOfFreedom._sel_list_to_positions( self.sels[key] )
        self.current_key = key
        ## previously:
        # result = [self._get_value(positions)]
        result = self._get_value(positions)
        assert( np.all( ~np.isnan(result) ) )
        assert( len(result) > 0 )
        self.current_key = None
        return result

    def compute_volume(self, bins):
        return np.diff(bins)

class ThreeDimensionalDensityDoF(DegreeOfFreedom):

    def __init__(self, *particles):
        # def __init__(self, origin, delta, num, *particles):
        # self.origin = origin    # 3d ndarray
        # self.delta = delta      # 3d ndarray
        # self.num = num          # 3d ndarray
        # self.center = self.origin + 0.5*self.delta*self.num
        DegreeOfFreedom.__init__(self, *particles)
        
    def _get_value(self, positions):
        return positions

    def compute_volume(self, bins):
        assert(len(bins) == 3)
        return np.prod([a[1]-a[0] for a in bins])
    
class RadiusDof(DegreeOfFreedom):
    def _get_value(self, positions):
        for p in positions:
            assert( len(p) == 3 )
        ri, = positions
        dist = np.linalg.norm(ri)
        return dist

    def compute_volume(self, bins):
         return (4.0/3)*np.pi*(bins[1:]**3-bins[:-1]**3)

class BondDof(DegreeOfFreedom):
    def _get_value(self, positions):
        for p in positions:
            assert( len(p) == 3 )
        ri,rj = positions
        dist = np.linalg.norm( self.wrap_vector(rj-ri) )
        return dist

    def compute_volume(self, bins):
         return (4.0/3)*np.pi*(bins[1:]**3-bins[:-1]**3)

class AngleDof(DegreeOfFreedom):
    def _get_value(self, positions):
        for p in positions:
            assert( len(p) == 3 )
        ri,rj,rk = positions
        rji = self.wrap_vector(ri-rj)
        rjk = self.wrap_vector(rk-rj)
        cos = rji.dot(rjk) / (np.linalg.norm(rji)*np.linalg.norm(rjk))
        angle = np.arccos(cos)
        angle = angle*180/np.pi
        return angle

    def compute_volume(self, bins):
        sin = np.sin(bins*np.pi/180)
        return 0.5*(sin[1:]+sin[:-1])

class DihedralDof(DegreeOfFreedom):
    def _get_value(self, positions):
        for p in positions:
            assert( len(p) == 3 )
        posa,posb,posc,posd = positions

        ab = self.wrap_vector(posa - posb)
        bc = self.wrap_vector(posb - posc)
        cd = self.wrap_vector(posc - posd)

        distbc = np.linalg.norm(bc)
        crossABC = np.cross(ab,bc)
        crossBCD = np.cross(bc,cd)
        crossX = np.cross(bc,crossABC)

        cos_phi = crossABC.dot(crossBCD) / (np.linalg.norm(crossABC) * np.linalg.norm(crossBCD))
        sin_phi = crossX.dot(crossBCD) / (np.linalg.norm(crossX) * np.linalg.norm(crossBCD))

        angle = -np.arctan2(sin_phi, cos_phi)
        angle = angle*180/np.pi
        return angle

class BondAngleDof(DegreeOfFreedom):
    def _get_value(self, positions):
        raise NotImplementedError

        for p in positions:
            assert( len(p) == 3 )
        posa,posb,posc,posd = positions

        ab = self.wrap_vector(posa - posb)
        bc = self.wrap_vector(posb - posc)
        cd = self.wrap_vector(posc - posd)

        distbc = np.linalg.norm(bc)
        crossABC = np.cross(ab,bc)
        crossBCD = np.cross(bc,cd)
        crossX = np.cross(bc,crossABC)

        cos_phi = crossABC.dot(crossBCD) / (np.linalg.norm(crossABC) * np.linalg.norm(crossBCD))
        sin_phi = crossX.dot(crossBCD) / (np.linalg.norm(crossX) * np.linalg.norm(crossBCD))

        angle = -np.arctan2(sin_phi, cos_phi)
        angle = angle*180/np.pi
        return angle

class PairDistributionDof():
    def __init__(self, particles_A, particles_B, range_=(0,50), resolution=0.1, exclusions=None):
        if exclusions is None: exclusions = []

        self.particles_A = particles_A
        self.particles_B = particles_B
        self.range_      = range_
        self.resolution  = resolution
        self.exclusions  = exclusions

        self.sel_A             = dict()
        self.sel_B             = dict()
        self.exclusions_mask   = dict()

    def _particles_to_sel(particles, universe):
        for particle in particles:
            if isinstance(particle, ArbdModel._GroupSite):
                raise NotImplementedError

        sel_string = f'index {" ".join([str(p.idx) for p in particles])}'
        return universe.select_atoms( sel_string )


    def _build_exclusion_matrix(self, universe):
        key = universe.__hash__()
        if key in self.exclusions_mask:
            return self.exclusions_mask[key]

        logger.info('Building exclusions')
        n_atoms = len(universe.atoms)

        A, B = self.sel_A[key], self.sel_B[key]

        ex = self.exclusions_mask[key] = ex = dict()

        _index_to_sel_order_A = -1*np.ones(n_atoms)
        _index_to_sel_order_B = -1*np.ones(n_atoms)
        for i,I in enumerate(A.atoms.indices):
            _index_to_sel_order_A[I] = i
            ex[(i,i)] = 1

        for i,I in enumerate(B.atoms.indices):
            _index_to_sel_order_B[I] = i
            ex[(i,i)] = 1

        for p1,p2 in self.exclusions:
            _ids = [(_index_to_sel_order_A[p1.idx], _index_to_sel_order_B[p2.idx]),
                    (_index_to_sel_order_A[p2.idx], _index_to_sel_order_B[p1.idx])]
            for i,j in _ids:
                if i < 0 or j < 0: continue
                ex[(i,j)] = ex[(j,i)] = 1

        logger.info('Done building exclusions')
        return ex

    def get_values(self, universe):
        from MDAnalysis.lib import distances

        key = universe.__hash__()
        if key not in self.sel_A:
            self.sel_A[key] = PairDistributionDof._particles_to_sel(self.particles_A, universe)
        if key not in self.sel_B:
            self.sel_B[key] = PairDistributionDof._particles_to_sel(self.particles_B, universe)
        A = self.sel_A[key]
        B = self.sel_B[key]

        _pairs,_dist = distances.capped_distance( A.positions, B.positions,
                                                self.range_[1] + self.resolution,
                                                box = universe.dimensions)
        # n_atoms = len(universe.atoms)
        ex = self._build_exclusion_matrix(universe)
        return [d for p,d in zip(_pairs,_dist) if tuple(p) not in ex]

    def compute_volume(self, bins):
        return (4.0/3)*np.pi*(bins[1:]**3-bins[:-1]**3)

class AbstractIBIpotential(AbstractPotential, metaclass=ABCMeta):
    """
    A class that implements the Iterative Boltzmann Inversion method for generating potentials.

    This abstract class provides the foundation for creating IBI potentials that are derived
    from target distributions. IBI is a method to derive effective pair potentials from radial
    distribution functions in molecular systems.

    Parameters
    ----------
    name : str
        Name of the potential, used for file naming.
    degrees_of_freedom : list
        List of degrees of freedom to consider in the potential.
    range_ : tuple, optional
        Range of distances to consider (min, max), default is (0, 30).
    resolution : float, optional
        Resolution of the binning for the distribution, default is 0.1.
    max_force : float, optional
        Maximum force allowed in the potential.
    max_potential : float, optional
        Maximum potential value allowed.
    out_of_bounds_force : str or float, optional
        Force to apply outside the well-defined density region, default is 'max_force'.
    zero : str, optional
        Method to set the zero point of the potential, default is 'last'.
    smooth : int or None, optional
        Window size for Savitzky-Golay filter, if None will be calculated automatically.
    learning_rate : float or callable, optional
        Learning rate for potential updates, default is 0.9.
    iteration : int, optional
        Current iteration number, default is 1.
    filename_prefix : str, optional
        Path prefix for saving potentials, default is 'IBIPotentials/'.

    Attributes
    ----------
    bins : numpy.ndarray
        Bin edges for the distribution.
    potential : numpy.ndarray
        Current potential values.

    Methods
    -------
    potential(r)
        Abstract method to calculate potential at distance r.
    filename(types=None, iteration=None, smoothed=True)
        Generate filename for the potential.
    write_file()
        Write the potential to a file.
    get_target_distribution(universe=None, trajectory=None, recalculate=False)
        Get or calculate the target distribution.
    get_cg_distribution(universe, trajectory=None, box=None, recalculate=False)
        Get or calculate the coarse-grained distribution.
    read_cg_potential(iteration=None)
        Read a potential from a file.
    write_cg_potential(universe=None, scaling_factor=None, temperature=295, tol=None, clean_edges=True, box=None)
        Calculate and save the potential based on distributions.

    Notes
    -----
    IBI is an iterative method that refines potentials to match target distributions.
    The process involves:
    1. Starting with an initial guess (usually Boltzmann inversion of the target)
    2. Running simulations with the current potential
    3. Comparing resulting distributions with the target
    4. Updating the potential based on the difference
    5. Repeating until convergence
    """
    def __init__(self, name, degrees_of_freedom=[], range_=(0,30), resolution=0.1, max_force=None, max_potential=None, out_of_bounds_force='max_force', zero='last', smooth=None, learning_rate=0.9, iteration=1, filename_prefix='IBIPotentials/'):
        self.name = name
        if degrees_of_freedom is None: degrees_of_freedom = []
        self.degrees_of_freedom = degrees_of_freedom
        self.smooth = smooth
        self.iteration = iteration

        AbstractPotential.__init__(self, range_, resolution, max_force, max_potential, zero)

        self.filename_prefix = filename_prefix + self.name
        self.max_potential = max_potential
        self.out_of_bounds_force = out_of_bounds_force
        self.iteration = iteration
        # self.smooth = 15 if smooth is None else smooth
        self.learning_rate = learning_rate

        self._target = None
        self.bins = np.arange( self.range_[0],
                               self.range_[1]+self.resolution,
                               self.resolution )
        self.potential = np.zeros( len(self.bins)-1 )
        self.__dists = {}

    def potential(self, r):
        raise NotImplementedError

        ## self.filename_prefix="IBIpotentials/"

    def filename(self, types=None, iteration=None, smoothed=True, directory='.'):
        if iteration is None:
            iteration = self.iteration
        return f"{directory}/{self.filename_prefix}-{iteration:03d}{'' if smoothed else '-raw'}.dat"

    def __str__(self):
        return self.filename()

    def write_file(self):
        pass

    def __hash__(self):
        assert(self.type_ != "None")
        return hash((self.name, self.range_, self.resolution, self.max_force, self.max_potential, self.out_of_bounds_force, self.filename_prefix, self.periodic))

    def __eq__(self, other):
        # def _get_attr_mangle(obj,a):
        #     try:
        #         v = obj.__dict__[a]
        #     except:
        #         v = obj.__dict__[f'_AbstractPotential__{a}']
        #     return v

        for a in ("name", "range_", "resolution", "max_force", "max_potential", "filename_prefix", "periodic"):
            # if _get_attr_mangle(self,a) != _get_attr_mangle(other,a):
            if getattr(self,a) != getattr(other,a):
                return False
        return True

    def bin_dof_data(self, bins, vals, counts):
        """ Function to bin position data from DoFs into 3D counts """
        if sum(np.array(counts.shape) > 1) > 1:
            _tmp, _ = np.histogramdd(vals, bins=bins, density=False)
        else:
            _tmp, _ = np.histogramdd(vals, bins=[bins], density=False)
        counts += _tmp

        # """ Function to bin data from DoFs into 1D counts """
        # inds = np.digitize(vals, bins) - 1
        # # if (inds < 0).sum() > 0: (inds >= len(counts)).sum() > 0:
        # #     logger.warn(f'inds contains {(inds < 0).sum()} elements < 0 and {(inds >= len(counts)).sum()} elements >= len(counts) ({len(counts)}) of {inds.size} total elements')
        # inds = inds[(inds<len(counts)) & (inds>=0)]
        # counts = counts + np.bincount(inds, minlength=len(counts))

    def symmetrize(self, bins, counts):
        """ Implement for subclasses as needed """
        pass
        
    def _extract_distribution(self, universe, trajectory = None, box = None):
        if trajectory is not None:
            key = (universe, trajectory).__hash__()
        else:
            key = universe.__hash__()
            trajectory = universe.trajectory

        if key in self.__dists:
            logger.info(f"{self}._extract_distribution( u.{key} ): using cache")
            return self.__dists[key]
        # devlogger.info(f"{self}._extract_distribution( u.{key} ): calculating")
        bins = self.bins
        try:
            counts = np.zeros( [len(a)-1 for a in bins] )
        except:
            counts = np.zeros( [len(bins)-1] )
        nframes = 0
        with logging_redirect_tqdm(loggers=[logger]):
            vals = []
            nvals = 0
            for t in tqdm(trajectory, desc=f"Extracting distribution {self}"):
                # if (t.frame % 100) == 0: logger.info(f'Calculating distribution associated with {self} {t.frame}/{len(universe.trajectory)-1}')
                if box is not None:
                    universe.dimensions = box
                _v = np.vstack([dof.get_values(universe) for dof in self.degrees_of_freedom])
                vals.append(_v)
                nvals += len(_v)
                if nvals > 10000000:
                    assert( vals[0].shape[1] == len(bins) )
                    self.bin_dof_data(bins, np.vstack(vals), counts)
                    vals = []
                    nvals = 0
                nframes += 1
            if nvals > 0:
                assert( vals[0].shape[1] == len(bins) )
                self.bin_dof_data(bins, np.vstack(vals), counts)
                vals = []
                nvals = 0
                
        if np.sum(counts) == 0:
            raise ValueError(f'Extracting distribution for "{self.filename()}" failed because no values were found in the range {self.range_}; consider specifying a larger range_ parameter when creating the {self.__class__.__name__}')
        assert(nframes > 0)

        self.symmetrize(bins,counts)
        
        vol = self.degrees_of_freedom[0].compute_volume(bins)
        assert(vol.sum() > 0)
        likelihood = counts / (nframes*vol)
        ## don't normalize over num values in dofs just yet
        # raise NotImplementedError("Why trim bins here?")
        # bins = bins[:len(likelihood)]
        self.__dists[key] = (bins,likelihood) # record for later
        return bins, likelihood

    def _save_distribution(self, filename, bins, vals):
        f = filename
        if np.sum(vals) == 0:
            raise Exception
        n_bin_arrs = len(bins)
        try: len(bins[0])
        except: n_bin_arrs = 1
        if n_bin_arrs > 1:
            bin_dict = {f'bins_{i}':b for i,b in enumerate(bins)}
        else:
            bin_dict = {'bins_0':bins}
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        counts = vals
        vals = vals / np.sum(vals)
        np.savez(f,n_bin_arrs=n_bin_arrs, **bin_dict, vals=vals, counts=counts, allow_pickle=False)
        return bins,vals,counts

    def _load_distribution(self, filename):
        f = filename
        _data = np.load(f, allow_pickle=False)
        n_bin_arrs, vals, counts = [_data[key] for key in ('n_bin_arrs', 'vals', 'counts')]
        bins = [_data[f'bins_{i}'] for i in range(n_bin_arrs)]
        if n_bin_arrs == 1: bins = bins[0]
        return bins,vals,counts
    
    def get_target_distribution(self, universe=None, trajectory=None, recalculate=False, directory='.'):
        if self._target is None:
            f = f'{directory}/{self.filename_prefix}.target.npz'
            if (not Path(f).exists()) or recalculate:
                if universe is None: raise Exception
                bins, vals = self._extract_distribution( universe, trajectory=trajectory )
                bins,vals,counts = self._save_distribution(f, bins, vals)
            else:
                bins,vals,counts = self._load_distribution(f)
            if np.sum(counts) == 0:
                raise Exception
            self._target = (bins,vals)

        if self.smooth is None:
            self.set_smooth_from_target_distribution()

        return self._target

    def set_smooth_from_target_distribution(self):
        bins, vals = self._target
        bins = 0.5*(bins[:-1] + bins[1:])
        ## Set smooth to ~1/2 stddev
        _mean = np.average( bins, weights=vals )
        _var = np.average( (bins-_mean)**2 , weights=vals )
        _dr = (bins[1]-bins[0])
        self.smooth = (int(np.round(np.sqrt(_var)/(2*_dr))+1)//2)*2+1
        if self.smooth < 5:
            logger.warning(f'Smoothing ({self.smooth} points) suggested by 1/2 of stddev ({np.sqrt(_var):02f}) is too low: setting smooth to 5')
            self.smooth = 5
        else:
            logger.info(f'Smoothing {self.smooth} points as suggested by 1/2 of stddev ({np.sqrt(_var):02f}) 5')

    def get_cg_distribution(self, universe, trajectory=None, box=None, recalculate=False, directory='.'):
        f = self.filename(smoothed=False,directory=directory)[:-3] + '.npz'

        if (not Path(f).exists()) or recalculate:
            logger.info(f"get_cg_distribution(): writing to '{f}'")
            bins, vals = self._extract_distribution( universe, trajectory=trajectory, box=box )
            bins, vals, counts = self._save_distribution(f, bins, vals)
        else:
            logger.info(f"get_cg_distribution(): reading from '{f}'")
            bins, vals, counts = self._load_distribution(f)
            key = universe.__hash__()
            if key not in self.__dists:
                self.__dists[key] = (bins, counts)
        return bins, vals

    def read_cg_potential(self, iteration=None, directory='.'):
        if iteration is None: iteration = self.iteration-1
        if iteration == 0:
            bins = self.bins
            bins = 0.5*(bins[:-1] + bins[1:])
            pot = np.zeros(bins.shape)
        else:
            try:
                f = self.filename(iteration=iteration, smoothed=True, directory=directory)
                bins, pot = np.loadtxt(f).T
            except:
                f = self.filename(iteration=iteration, smoothed=False, directory=directory)
                bins, pot = np.loadtxt(f).T
        return bins,pot

    # def _clean_edges(self, bins, rho, tol):

    #     """ Removes values to the left of the rightmost spot left of
    #     the peak where rho dips below tol, and vice versa """

    #     total_before = np.sum(rho)
    #     peak_i = np.where(np.abs(rho) == np.max(np.abs(rho)))[0][0]
    #     is_left  = (bins < bins[peak_i])
    #     is_small = (rho <= tol)
    #     try:
    #         first_i = np.where(is_left & is_small)[0][-1]
    #     except:
    #         first_i = 0
    #     try:
    #         last_i = np.where((~is_left) & is_small)[0][0]
    #     except:
    #         last_i = len(rho)

    #     rho[:first_i] = 0
    #     rho[last_i:] = 0

    #     total_after = np.sum(rho)
    #     if total_after < 0.9 * total_before:
    #         raise ValueError('Removed too much density from the distribution ({100*total_after/total_before:%02d})')

    def _clean_edges(self, bins, rho, tol):
        """ Removes values outside the largest island of density """
        total_before = np.sum(rho)
        struc = np.ones( [3 for b in bins], dtype=int )
        l,num_l = label( rho > tol, structure=struc )
        _u,_counts = np.unique(l,return_counts=True)
        _counts[_u == 0] = 0    # ensure we don't select (rho < tol) as largest island
        valid = (l == np.argmax(_counts))
        
        logger.info(f'_clean_edges: Removing {100*rho[~valid].sum() / rho.sum():.1f}% of density')
        logger.info(tol)
        # import ipdb; ipdb.set_trace()

        # rho[~valid] = 0
        # rho[~valid] = tol

        total_after = np.sum(rho)
        if total_after < 0.9 * total_before:
            raise ValueError(f'Removed too much density from the distribution ({100*total_after/total_before:02f})')

    def apply_out_of_bounds_force(self, u, valid):
        ## Apply boundary force outside where target density is well-defined
        oobf = self.max_force if self.out_of_bounds_force == 'max_force' else self.out_of_bounds_force

        bins = self.bins
        delta = [_b[1]-_b[0] for _b in bins]
        if oobf is None:
            oobf = 2 * max(
                np.max(np.abs( np.diff(u,axis=i) / _d ))
                for i,_d in enumerate(delta))
            if self.out_of_bounds_force == 'max_force':
                logger.warning(f'IBI out-of-bounds force for {self} was set to max_force, but max_force was unspecified; defaulting to twice the largest value found in valid range ({oobf:.2f}) and setting the value permanently.')
                self.out_of_bounds_force = oobf

        valid_voxel_coords = np.argwhere(valid)
        if valid_voxel_coords.size < 3*valid.size:
            # raise Exception('Obsolete: use `ndimage.morphology.distance_transform_edt( boolean_grid, sampling=[Dxy,Dxy,Dz] )`')
            logger.info('Building OOBF KDtree')
            tree = KDTree([r * delta for r in valid_voxel_coords])
            invalid_voxel_coords = np.argwhere(~valid)
            logger.info('Filling invalid potential values from OOBF KDtree')
            for r in invalid_voxel_coords:
                dist,idx = tree.query(r*delta)
                u[tuple(r)] = u[tuple(valid_voxel_coords[idx])] + oobf*dist
            logger.info('Done filling invalid potential values from OOBF KDtree')
        
    def write_cg_potential(self, universe=None, scaling_factor = None, temperature = 295, tol = None, clean_edges=True, box=None, directory='.'):
        if scaling_factor is None:
            try:    scaling_factor = self.learning_rate(self.iteration)
            except: scaling_factor = self.learning_rate

        savgol_opts = dict(
            window_length=1+(self.smooth//2)*2, polyorder=3,
            mode = 'wrap' if self.periodic else 'nearest'
        )

        bins_aa, rho_aa = self.get_target_distribution(directory=directory)
        bins_aa = 0.5*(bins_aa[:-1] + bins_aa[1:])
        rho_aa = rho_aa / np.sum(rho_aa)

        if tol is None:
            tol = max(1e-5, 1e-3 * np.max(rho_aa)) # likely there is room for improvement here

        if universe is None:
            bins = 0.5*(self.bins[:-1] + self.bins[1:])
            assert( np.all(np.isclose(bins - bins_aa, 0)) )

            if clean_edges:
               self._clean_edges(bins_aa, rho_aa, tol)

            rho_aa = savgol( rho_aa, **savgol_opts )
            rho_aa[rho_aa < tol] = tol

            ## units "295 k K" kcal_mol
            u = - scaling_factor * 0.58622592 * (temperature/295) * np.log(rho_aa)
            rho_cg = rho_aa     # allows a common smoothing command below
        else:
            bins, rho_cg = self._extract_distribution( universe, box=box )
            bins = 0.5*(bins[:-1] + bins[1:])
            assert( np.abs(len(rho_cg) - len(bins)) < 1 )
            rho_cg = rho_cg/np.sum(rho_cg)
            assert( np.all(np.isclose(bins - bins_aa, 0)) )

            if clean_edges:
                self._clean_edges(bins_aa, rho_aa, tol)
                # self._clean_edges(bins_aa, rho_cg, tol)


            r0,u0 = self.read_cg_potential(directory=directory) # iteration-2?
            rho_cg,rho_aa = [savgol( rho, **savgol_opts ) for rho in [rho_cg,rho_aa]]
            rho_aa[rho_aa < tol] = tol
            rho_cg[rho_cg < tol] = tol

            ratio = rho_cg/rho_aa
            ## units "295 k K" "kcal_mol"
            du = np.log( ratio )
            du = du * 0.58622592 * (temperature/295)
            du = du * (rho_aa/rho_aa.max())**0.25 # penalize learning for values where target density is very low

            try:    alpha = self.learning_rate(self.iteration)
            except: alpha = self.learning_rate
            u = u0 + alpha * du

        f = self.filename(smoothed=False, directory=directory)
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(f,np.array((bins,u)).T)

        f = self.filename(smoothed=True, directory=directory)

        ## Only apply savgol filter in region where target density is well-defined
        valid = np.where(rho_aa > tol)[0]
        first = valid[0]
        last = valid[-1]
        u[first:last+1] = savgol( u[first:last+1], **savgol_opts )

        # ## Apply boundary force outside where target density is well-defined
        # oobf = self.max_force if self.out_of_bounds_force == 'max_force' else self.out_of_bounds_force
        # if oobf is None:
        #     oobf = 2 * np.max(np.abs( np.diff(u[first:last+1]) / (bins[first+1]-bins[first]) ))
        #     if self.out_of_bounds_force == 'max_force':
        #         logger.warning(f'IBI out-of-bounds force for {self} was set to max_force, but max_force was unspecified; defaulting to twice the largest value found in valid range ({oobf:%.2f}) and setting the value permanently.')
        #         self.out_of_bounds_force = oobf

        # if first > 0:
        #     u[:first] = u[first] + np.abs(bins[:first]-bins[first])*oobf
        # if last < len(u)-2:
        #     u[last+1:] = u[last] + np.abs(bins[last+1:]-bins[last])*oobf

        self.apply_out_of_bounds_force(u,valid)
            
        u = self._cap_potential(bins, u)
        np.savetxt(f,np.array((bins,u)).T)
        return bins,u

## Specialize with sensible defaults for r_range
class IBIBond(AbstractIBIpotential):
    def __init__(self, name, degrees_of_freedom=[], range_=(0,20), resolution=0.02, max_force=None, max_potential=None, out_of_bounds_force='max_force', zero='min', smooth=None, learning_rate=0.9, iteration=1, filename_prefix="IBIPotentials/"):
        # AbstractIBIpotential.__init__(self, name, degrees_of_freedom, range_, resolution, smooth, iteration, max_force, max_potential)
        AbstractIBIpotential.__init__(self, name, degrees_of_freedom, range_, resolution, max_force, max_potential, out_of_bounds_force, zero, smooth, learning_rate, iteration, filename_prefix)
        self.type_ = 'IBIbond'

class IBIAngle(AbstractIBIpotential):
    def __init__(self, name, degrees_of_freedom, range_=(0,180), resolution=2.0, max_force=None, max_potential=None, out_of_bounds_force='max_force', zero='min', smooth=None, learning_rate=0.9, iteration=1, filename_prefix="IBIPotentials/"):
        #rm: AbstractIBIpotential.__init__(self, name, degrees_of_freedom, range_, resolution, smooth, iteration, max_force, max_potential, iteration, filename_prefix)
        AbstractIBIpotential.__init__(self, name, degrees_of_freedom, range_, resolution, max_force, max_potential, out_of_bounds_force, zero, smooth, learning_rate, iteration, filename_prefix)
        self.type_ = 'IBIangle'

class IBIDihedral(AbstractIBIpotential):
    def __init__(self, name, degrees_of_freedom=[], range_=(-180,180), resolution=4.0, max_force=None, max_potential=None, out_of_bounds_force='max_force', zero='min', smooth=None, learning_rate=0.9, iteration=1, filename_prefix="IBIPotentials/"):
        #rm: AbstractIBIpotential.__init__(self, name, degrees_of_freedom, range_, resolution, smooth, iteration, max_force, max_potential, filename_prefix)
        AbstractIBIpotential.__init__(self, name, degrees_of_freedom, range_, resolution, max_force, max_potential, out_of_bounds_force, zero, smooth, learning_rate, iteration, filename_prefix)
        self.type_ = 'IBIdihed'

    @property
    def periodic(self):
        return True

## Specialize with sensible defaults for r_range
class IBINonbonded(AbstractIBIpotential):
    def __init__(self, name, degrees_of_freedom=[], range_=(0,50), resolution=0.1, max_force=None, max_potential=None, out_of_bounds_force='max_force', zero='last', smooth=None, learning_rate=0.35, iteration=1, filename_prefix="IBIPotentials/"):
        AbstractIBIpotential.__init__(self, name, degrees_of_freedom, range_, resolution, max_force, max_potential, out_of_bounds_force, zero, smooth, learning_rate, iteration, filename_prefix)
        self.type_ = 'IBInonbonded'

    def write_file(self, filename=None, types=None):
        if filename is None:
            filename = self.filename(types=types)
        assert( filename == self.filename(types=types) )

    
class IBIGrid(AbstractIBIpotential):
    """ Grid potential """

    def __init__(self, name, origin, delta, num_voxels, degrees_of_freedom=[], max_force=None, max_potential=None, out_of_bounds_force='max_force', zero='last', smooth=None, learning_rate=0.35, iteration=1, filename_prefix="IBIPotentials/", write_density=False):
        range_ = (0,1)
        resolution = 1
        AbstractIBIpotential.__init__(self, name, degrees_of_freedom, range_, resolution, max_force, max_potential, out_of_bounds_force, zero, smooth, learning_rate, iteration, filename_prefix)
        self.origin = origin
        self.delta = delta        
        self.num_voxels = num_voxels
        self.type_ = 'IBIGrid'
        self.bins = [np.arange(n)*d+o for n,d,o in zip(num_voxels,delta,origin)]
        self.write_density = write_density
        
    def write_file(self, filename=None, types=None):
        if filename is None:
            filename = self.filename(types=types)
        assert( filename == self.filename(types=types) )

    def set_smooth_from_target_distribution(self):
        bins,vals = self._target

        self.smooth = []
        for i,_b in enumerate(bins):
            _v = vals.sum( axis = tuple(j for j in range(len(bins)) if j != i) )
            _b = 0.5*(_b[:-1] + _b[1:])
            ## Set smooth to ~1/2 stddev
            _mean = np.average( _b, weights=_v  )
            _var = np.average( (_b-_mean)**2 , weights=_v )
            _dr = (_b[1]-_b[0])
            s = int(np.round(np.sqrt(_var)/(2*_dr))+1)//2*2+1
            if s < 5:
                logger.warning(f'Smoothing along axis {i} ({s} points) suggested by 1/2 of stddev ({np.sqrt(_var):02f}) is too low: setting smooth to 5')
                s = 5
            else:
                logger.info(f'Smoothing {s} points as suggested by 1/2 of stddev ({np.sqrt(_var):02f}) 5')
            self.smooth.append(s)

    def __remove_nans(self, u):
        points = np.stack( np.meshgrid(*[b[:-1] for b in self.bins],indexing='ij' ),axis=-1).reshape(-1,3)
        assert( len(points) == u.size )
        nans=np.isnan(u.flat)
        if nans.sum() > 0:
            u[nans] = RBFInterpolator(points[~nans], u.flat[~nans], smoothing=0.5, neighbors=27)(points[nans])

    def _cap_potential(self, r, u):
        self.__remove_nans(u)

        if self.zero == 'min':
            u = u - np.min(u)
        elif self.zero == 'last':
            _m = np.array([u[tuple(slice(idx,idx+1) if sl_i == i else slice(None) for i in range(len(u.shape)))].mean() for idx in (0,-2) for sl_i in range(len(u.shape))]).mean()
            u = u - _m
        else:
            raise ValueError('Unrecognized option for "zero" argument')
        
        max_force = self.max_force
        if max_force is not None:
            raise NotImplementedError
            # assert(max_force > 0)
            # f = np.diff(u)/np.diff(r)
            # f[f>max_force] = max_force
            # f[f<-max_force] = -max_force
            # u[0] = 0
            # u[1:] = np.cumsum(f*np.diff(r))

        if self.max_potential is not None:
            sl = u > self.max_potential
            u[sl] = self.max_potential

        if self.zero == 'min':
            u = u - np.min(u)
        elif self.zero == 'last':
            _m = np.array([u[tuple(slice(idx,idx+1) if sl_i == i else slice(None) for i in range(len(u.shape)))].mean() for idx in (0,-2) for sl_i in range(len(u.shape))]).mean()
            u = u - _m
        else:
            raise ValueError('Unrecognized option for "zero" argument')
        return u

    def filename(self, types=None, iteration=None, smoothed=True, directory='.'):
        return super().filename(types, iteration, smoothed, directory)[:-4]+'.dx'

    def read_cg_potential(self, iteration=None, directory='.'):
        if iteration is None: iteration = self.iteration-1
        if iteration == 0:
            pot = np.zeros( np.prod([len(a)-1 for a in bins]) )
            raise NotImplementedError
            bins = [a[:-1] for a in bins]
        else:
            try:
                f = self.filename(iteration=iteration, smoothed=True, directory=directory)
                g = Grid(f)
            except:
                f = self.filename(iteration=iteration, smoothed=False, directory=directory)
                g = Grid(f)
            pot = g.grid
            bins = [a[:-1] for a in g.edges] 
        return bins,pot

    def write_cg_potential(self, universe=None, scaling_factor = None, temperature = 295, tol = None, clean_edges=True, box=None, directory='.'):
        if scaling_factor is None:
            try:    scaling_factor = self.learning_rate(self.iteration)
            except: scaling_factor = self.learning_rate

        smooth_opts = dict(
            window_length=[1+(s//2)*2 for s in self.smooth],
            polyorder = 3,
            mode = 'wrap' if self.periodic else 'nearest'
        )

        bins_aa, rho_aa = self.get_target_distribution(directory=directory)
        f = self.filename(smoothed=False, directory=directory)
        if self.iteration == 1:
            writeDx( f+'.target.dx', rho_aa, self.origin, self.delta, fmt='%.6f')

        rho_aa = rho_aa / np.sum(rho_aa)
        
        if tol is None:
            tol = max(10**(-5*len(bins_aa)), 10**(-3*len(bins_aa))*np.max(rho_aa)) # likely there is room for improvement here

        if universe is None:
            # raise NotImplementedError('Unsure about trimming bins')
            # bins = self.bins[:-1]
            bins = self.bins
            assert( all(np.all(np.isclose(b1 - b2, 0)) for b1,b2 in zip(bins,bins_aa)) )

            if clean_edges:
               self._clean_edges(bins_aa, rho_aa, tol)

            if np.prod(smooth_opts['window_length']) > 1:
                rho_aa = savgol_nd( rho_aa, **smooth_opts )
            rho_aa[rho_aa < tol] = tol

            ## units "295 k K" kcal_mol
            u = - scaling_factor * 0.58622592 * (temperature/295) * np.log(rho_aa)
            rho_cg = rho_aa     # allows a common smoothing command below
        else:
            bins, rho_cg = self._extract_distribution( universe, box=box )
            # assert( np.abs(len(rho_cg) - len(bins)) < 1 )
            if self.write_density:
                writeDx( f+'.cgden.raw.dx', rho_cg, self.origin, self.delta, fmt='%.6f')
            rho_cg = rho_cg/np.sum(rho_cg)
            assert( all(np.all(np.isclose(b1 - b2, 0)) for b1,b2 in zip(bins,bins_aa)) )

            if clean_edges:
                self._clean_edges(bins_aa, rho_aa, tol)
                # self._clean_edges(bins_aa, rho_cg, tol)

            r0,u0 = self.read_cg_potential(directory=directory)
            if np.prod(smooth_opts['window_length']) > 1:
                rho_cg,rho_aa = [savgol_nd( rho, **smooth_opts ) for rho in [rho_cg,rho_aa]]
            rho_aa[rho_aa < tol] = tol
            rho_cg[rho_cg < tol] = tol
            
            if self.iteration == 1:
                writeDx( f+'.target.dx', rho_aa, self.origin, self.delta, fmt='%.6f')
            if self.write_density:
                writeDx( f+'.cgden.dx', rho_aa, self.origin, self.delta, fmt='%.6f')

            ratio = rho_cg/rho_aa
            ## units "295 k K" "kcal_mol"
            du = np.log( ratio )
            du = du * 0.58622592 * (temperature/295)
            du = du * (rho_aa/rho_aa.max())**0.25 # penalize learning for values where target density is very low

            try:    alpha = self.learning_rate(self.iteration)
            except: alpha = self.learning_rate
            u = u0 + alpha * du

        f = self.filename(smoothed=False, directory=directory)
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        writeDx( f, u, self.origin, self.delta, fmt='%.6f')

        f = self.filename(smoothed=True, directory=directory)

        ## Apply savgol filter in region where target density is well-defined
        valid = (rho_aa > tol)
        if np.prod(smooth_opts['window_length']) > 1:
            u[valid] = savgol_nd( u, **smooth_opts )[valid]

        self.apply_out_of_bounds_force(u,valid)

        u = self._cap_potential(bins, u)
        writeDx( f, u, self.origin, self.delta, fmt='%.6f')

def writeDx(outfile, data, origin, delta, fmt="%.12f"):
  shape = np.shape(data)
  num = np.prod(shape)
  assert( len(shape) == 3 )
  assert( len(origin) == 3 )
  assert( len(delta) == 3 )
  try:
    assert( len(delta[0]) == 3 )
    dx,dy,dz = [" ".join(map(str,v)) for v in delta]
  except:
    dx = " ".join(map(str,[delta[0], 0, 0]))
    dy = " ".join(map(str,[0, delta[1], 0]))
    dz = " ".join(map(str,[0, 0, delta[2]]))
  headerInfo = dict( nx=shape[0], ny=shape[1], nz=shape[2],
                     ox=origin[0], oy=origin[1], oz=origin[2],
                     dx=dx, dy=dy, dz=dz,
                     num=num
                   )
  data = data.flatten(order='C')
  header = """# OpenDX density file
# File format: http://opendx.sdsc.edu/docs/html/pages/usrgu068.htm#HDREDF
object 1 class gridpositions counts  {nx} {ny} {nz}
origin {ox} {oy} {oz}
delta  {dx}
delta  {dy}
delta  {dz}
object 2 class gridconnections counts  {nx} {ny} {nz}
object 3 class array type double rank 0 items {num} data follows""".format(**headerInfo)
  len(data)

  if num == 3*(num//3):
    footer = ""
  else:
    footer = " ".join([fmt % x for x in data[3*(num//3):]]) # last line of data
    footer += "\n"

  footer += """attribute "dep" string "positions"
object "density" class field 
component "positions" value 1
component "connections" value 2
component "data" value 3
"""
  np.savetxt( outfile, np.reshape(data[:3*(num//3)], (num//3,3), order='C'), 
              fmt=fmt,
              header=header, comments='', footer=footer )
