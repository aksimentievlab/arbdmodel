from abc import ABCMeta
from scipy.signal import savgol_filter as savgol
import numpy as np
from pathlib import Path
from . import ArbdModel, logger
from .interactions import AbstractPotential

from tqdm import tqdm,trange
from tqdm.contrib.logging import logging_redirect_tqdm


""" This file includes various routines to simplify running
simulations with Iterative Boltmann Inversion. """


class DegreeOfFreedom():
    """ Base class for representing a degree of freedom in the system.

    Concrete examples include the distance, angle, or dihedral angle,
 between nearby "bonded" interaction sites, or the distances between
 pairs of particle types (with exclusions!).

    Derived classes should calculate a value for the DoF from particle
  coordinates.  """


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
        box = self.sels[self.current_key][0].universe.dimensions
        assert(np.all( np.array(box[3:]) == 90 ))

        for i in range(3):
            sl = (v[...,i] > 0.5*box[i])
            v[sl,i] = v[sl,i] - box[i]

            sl = (v[...,i] < -0.5*box[i])
            v[sl,i] = v[sl,i] + box[i]
        return v

    def get_values(self, universe):
        key = universe.__hash__()
        if key not in self.sels:
            self.sels[key] = [DegreeOfFreedom._particle_to_sel(p,universe)
                              for p in self.particles]
        positions = DegreeOfFreedom._sel_list_to_positions( self.sels[key] )
        self.current_key = key
        result = [self._get_value(positions)]
        assert( np.all( ~np.isnan(result) ) )
        assert( len(result) > 0 )
        self.current_key = None
        return result

    def compute_volume(self, bins):
        return np.diff(bins)

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
    def __init__(self, name, degrees_of_freedom=None, range_=(0,30), resolution=0.1, max_force=None, max_potential=None, out_of_bounds_force='max_force', zero='last', smooth=None, learning_rate=0.9, iteration=1, filename_prefix='IBIPotentials/'):
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

        self.__target = None
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
        counts = np.zeros( [len(bins)-1] )
        nframes = 0
        with logging_redirect_tqdm(loggers=[logger]):
            for t in tqdm(trajectory, desc=f"Extracting distribution {self}"):
                # if (t.frame % 100) == 0: logger.info(f'Calculating distribution associated with {self} {t.frame}/{len(universe.trajectory)-1}')
                if box is not None:
                    universe.dimensions = box
                vals = np.stack([dof.get_values(universe) for dof in self.degrees_of_freedom])
                inds = np.digitize(vals, bins) - 1
                # if (inds < 0).sum() > 0: (inds >= len(counts)).sum() > 0:
                #     logger.warn(f'inds contains {(inds < 0).sum()} elements < 0 and {(inds >= len(counts)).sum()} elements >= len(counts) ({len(counts)}) of {inds.size} total elements')
                inds = inds[(inds<len(counts)) & (inds>=0)]
                counts = counts + np.bincount(inds, minlength=len(counts))
                nframes += 1
        if np.sum(counts) == 0:
            raise ValueError(f'Extracting distribution for "{self.filename()}" failed because no values were found in the range {self.range_}; consider specifying a larger range_ parameter when creating the {self.__class__.__name__}')
        assert(nframes > 0)

        vol = self.degrees_of_freedom[0].compute_volume(bins)
        assert(vol.sum() > 0)
        likelihood = counts / (nframes*vol)
        ## don't normalize over num values in dofs just yet

        self.__dists[key] = (bins,likelihood) # record for later
        return bins, likelihood

    def get_target_distribution(self, universe=None, trajectory=None, recalculate=False, directory='.'):
        if self.__target is None:
            f = f'{directory}/{self.filename_prefix}.target.dat'
            if (not Path(f).exists()) or recalculate:
                if universe is None: raise Exception
                bins, vals = self._extract_distribution( universe, trajectory=trajectory )
                if np.sum(vals) == 0:
                    raise Exception
                Path(f).parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(f,np.array((bins[:-1],vals/np.sum(vals),vals)).T)
            bins, vals, counts = np.loadtxt(f).T
            if np.sum(counts) == 0:
                raise Exception
            self.__target = (bins,vals)
        if self.smooth is None:
            bins,vals = self.__target
            ## Set smooth to ~1/2 stddev
            _mean = np.average( bins, weights=vals )
            _var = np.average( (bins-_mean)**2 , weights=vals )
            _dr = (bins[1]-bins[0])
            self.smooth = (int(np.round(np.sqrt(_var)/(2*_dr))+1)//2)*2+1
            if self.smooth < 5:
                logger.warning(f'{f}: Smoothing ({self.smooth} points) suggested by 1/2 of stddev ({np.sqrt(_var):02f}) is too low: setting smooth to 5')
                self.smooth = 5
            else:
                logger.info(f'{f}: Smoothing {self.smooth} points as suggested by 1/2 of stddev ({np.sqrt(_var):02f}) 5')

        return self.__target

    def get_cg_distribution(self, universe, trajectory=None, box=None, recalculate=False, directory='.'):
        f = self.filename(smoothed=False,directory=directory).replace('.dat','.cg.dat')

        if (not Path(f).exists()) or recalculate:
            logger.info(f"get_cg_distribution(): writing to '{f}'")
            bins, vals = self._extract_distribution( universe, trajectory=trajectory, box=box )
            bins = bins[:len(vals)]
            Path(f).parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(f,np.array((bins,vals/np.sum(vals),vals)).T)
        else:
            logger.info(f"get_cg_distribution(): reading from '{f}'")
            bins, vals, counts = np.loadtxt(f).T
            key = universe.__hash__()
            if key not in self.__dists:
                self.__dists[key] = (bins, counts)
        return bins, vals

    def read_cg_potential(self, iteration=None, directory=directory):
        if iteration is None: iteration = self.iteration-1
        if iteration == 0:
            bins = self.bins[:-1]
            pot = np.zeros(bins.shape)
        else:
            try:
                f = self.filename(iteration=iteration, smoothed=True, directory=directory)
                bins, pot = np.loadtxt(f).T
            except:
                f = self.filename(iteration=iteration, smoothed=False, directory=directory)
                bins, pot = np.loadtxt(f).T
        return bins,pot


    def _apply_max_force(self, bins, u, rho, tol, savgol_opts):
        if self.max_force is not None:
            valid = np.where(rho > tol)[0]
            first = valid[0]
            last = valid[-1]
            if first > 0:
                u[:first] = u[first] + np.abs(bins[:first]-bins[first])*self.max_force
            if last < len(u)-2:
                u[last+1:] = u[last] + np.abs(bins[last+1:]-bins[last])*self.max_force
        return u

    def _clean_edges(self, bins, rho, tol):

        """ Removes values to the left of the rightmost spot left of
        the peak where rho dips below tol, and vice versa """

        total_before = np.sum(rho)
        peak_i = np.where(np.abs(rho) == np.max(np.abs(rho)))[0][0]
        is_left  = (bins < bins[peak_i])
        is_small = (rho <= tol)
        try:
            first_i = np.where(is_left & is_small)[0][-1]
        except:
            first_i = 0
        try:
            last_i = np.where((~is_left) & is_small)[0][0]
        except:
            last_i = len(rho)

        rho[:first_i] = 0
        rho[last_i:] = 0

        total_after = np.sum(rho)
        if total_after < 0.9 * total_before:
            raise ValueError('Removed too much density from the distribution ({100*total_after/total_before:%02d})')

    def write_cg_potential(self, universe=None, scaling_factor = None, temperature = 295, tol = None, clean_edges=True, box=None, directory='.'):
        if scaling_factor is None:
            try:    scaling_factor = self.learning_rate(self.iteration)
            except: scaling_factor = self.learning_rate

        savgol_opts = dict(
            window_length=1+(self.smooth//2)*2, polyorder=3,
            mode = 'wrap' if self.periodic else 'nearest'
        )

        bins_aa, rho_aa = self.get_target_distribution(directory=directory)
        rho_aa = rho_aa / np.sum(rho_aa)

        if tol is None:
            tol = max(1e-5, 1e-3 * np.max(rho_aa)) # likely there is room for improvement here

        if universe is None:
            bins = self.bins[:-1]
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
            assert( np.abs(len(rho_cg) - len(bins)) < 2 )
            bins = bins[:len(rho_cg)]
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

        ## Apply boundary force outside where target density is well-defined
        oobf = self.max_force if self.out_of_bounds_force == 'max_force' else self.out_of_bounds_force
        if oobf is None:
            oobf = 2 * np.max(np.abs( np.diff(u[first:last+1]) / (bins[first+1]-bins[first]) ))
            if self.out_of_bounds_force == 'max_force':
                logger.warning(f'IBI out-of-bounds force for {self} was set to max_force, but max_force was unspecified; defaulting to twice the largest value found in valid range ({oobf:%.2f}) and setting the value permanently.')
                self.out_of_bounds_force = oobf

        if first > 0:
            u[:first] = u[first] + np.abs(bins[:first]-bins[first])*oobf
        if last < len(u)-2:
            u[last+1:] = u[last] + np.abs(bins[last+1:]-bins[last])*oobf

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
