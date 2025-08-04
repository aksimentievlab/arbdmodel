from abc import abstractmethod, ABCMeta
import os, sys
import scipy
import numpy as np
from shutil import copyfile
from .grid import writeDx
""" Module providing classes used to describe potentials in ARBD """


## Abstract classes that others inherit from
class AbstractPotential(metaclass=ABCMeta):
    """
    Abstract base class for implementing interaction potentials.
    This class defines the basic structure for creating and managing interaction potential
    functions that can be written to files and used in simulations.

    Parameters
    ----------
    range_ : tuple of (float, float or None)
        The range of distances for the potential (min, max).
        If max is None, potential extends to infinity.
    resolution : float, optional
        The spacing between points in the discretized potential. Default is 0.1.
    max_force : float or None, optional
        Maximum allowed force magnitude. If provided, forces exceeding this value will be capped.
    max_potential : float or None, optional
        Maximum allowed potential energy. If provided, the potential will be modified
        to ensure it doesn't exceed this value.
    zero : str, optional
        Method to set the zero of the potential. Options are:
        - 'min': Zero is at the minimum value
        - 'first': Zero is at the first point
        - 'last': Zero is at the last point
    periodic : bool
        Whether the potential is periodic (False by default).

    Note
    -----
    Subclasses must implement at minimum the `potential` and `filename` methods.
    """
    """ Abstract class for writing potentials """

    def __init__(self, range_=(0,None), resolution=0.1, 
                 max_force = None, max_potential = None, zero='last'):
        self.range_ = range_
        self.resolution = resolution
        self.max_force = max_force
        self.max_potential = max_potential
        self.zero = zero

    @property
    def range_(self):
        return self.__range_
    @range_.setter
    def range_(self,value):
        assert(len(value) == 2)
        self.__range_ = tuple(value)

    @property
    def periodic(self):
        return False

    @abstractmethod
    def potential(self, r, types=None):
        raise NotImplementedError

    def __remove_nans(self, u):
        z = lambda x: np.argwhere(x).flatten()
        nans=np.isnan(u)
        u[nans] = scipy.interpolate.interp1d(z(~nans), u[~nans], fill_value='extrapolate')(z(nans))

    def __remove_nans_bak(self, u):
        left = np.isnan(u[:-1])
        right = np.where(left)[0]+1
        while np.any(np.isnan(u[right])):
            right[np.isnan(u[right])] += 1
        u[:-1][left] = u[right]

    def filename(self, types=None):
        raise NotImplementedError('Inherited potential objects should overload this function')

    def _cap_potential(self, r, u):
        self.__remove_nans(u)

        if self.zero == 'min':
            u = u - np.min(u)
        elif self.zero == 'first':
            u = u - u[0]
        elif self.zero == 'last':
            u = u - u[-1]
        else:
            raise ValueError('Unrecognized option for "zero" argument')
        
        max_force = self.max_force
        if max_force is not None:
            assert(max_force > 0)
            f = np.diff(u)/np.diff(r)
            f[f>max_force] = max_force
            f[f<-max_force] = -max_force
            u[0] = 0
            u[1:] = np.cumsum(f*np.diff(r))

        if self.max_potential is not None:
            f = np.diff(u)/np.diff(r)
            ids = np.where( 0.5*(u[1:]+u[:-1]) > self.max_potential )[0]

            # w = np.sqrt(2*self.max_potential/self.k)
            # drAvg = 0.5*(np.abs(dr[ids]) + np.abs(dr[ids+1]))
            # f[ids] = f[ids] * np.exp(-(drAvg-w)/(w))
            f[ids] = 0
            
            u[0] = 0
            u[1:] = np.cumsum(f*np.diff(r))

        if self.zero == 'min':
            u = u - np.min(u)
        elif self.zero == 'first':
            u = u - u[0]
        elif self.zero == 'last':
            u = u - u[-1]
        else:
            raise ValueError('Unrecognized option for "zero" argument')
        return u

    def write_file(self, filename=None, types=None):
        """
        Write potential energy values as a function of distance to a file.

        This method evaluates the potential function over a range of distances and writes
        the distance-potential pairs to a text file.

        Parameters
        ----------
        filename : str, optional
            The path to the output file. If None, a default filename will be generated using
            the `filename` method with the given types.
        types : tuple or list, optional
            The particle types for which to calculate the potential. Used for both filename 
            generation (if filename is None) and potential evaluation. If None, defaults to 
            the types defined in the interaction model.

        Note
        -----
        The potential values are capped using the `_cap_potential` method to handle
        potential singularities or extreme values. The output file contains two columns:
        distance (r) and potential energy (u).

        Any 'divide by zero' or 'invalid value' numpy warnings are ignored during
        the potential calculation.
        """
        if filename is None:
            filename = self.filename(types)
        rmin,rmax = self.range_
        r = np.arange(rmin, rmax+self.resolution, self.resolution)
        with np.errstate(divide='ignore',invalid='ignore'):
            u = self.potential(r, types)

        u = self._cap_potential(r,u)

        np.savetxt(filename, np.array([r,u]).T)

    def __hash__(self):
        try:    fname = self.filename(None)
        except: fname = self.filename(('',''))
        return hash((fname, self.range_, self.resolution, self.max_force, self.max_potential, self.zero))

    def __eq__(self, other):
        for a in ("range_", "resolution", "max_force", "max_potential", "zero"):
            if getattr(self,a) != getattr(other,a):
                return False
        if (type(self).__name__ != type(other).__name__):
            return False
        try:    fname1 = self.filename(None)
        except: fname1 = self.filename(('',''))
        try:    fname2 = self.filename(None)
        except: fname2 = self.filename(('',''))
        return  fname1 == fname2


    ## Want the same objects after copy operations
    def __copy__(self):
        return self
    def __deepcopy__(self, memo):
        return self
    
## Concrete nonbonded pontentials
class LennardJones(AbstractPotential):
    def potential(self, r, types):
        typeA, typeB = types
        epsilon = np.sqrt( typeA.epsilon**2 + typeB.epsilon**2 )
        r0 = 0.5 * (typeA.radius + typeB.radius)
        r6 = (r0/r)**6
        r12 = r6**2
        u = 4 * epsilon * (r12-r6)
        # u[0] = u[1]             # Remove NaN
        return u

class HalfHarmonic(AbstractPotential):
    def potential(self, r, types):
        typeA, typeB = types
        k = 10                   # kcal/mol AA**2
        r0 = (typeA.radius + typeB.radius)
        u =  0.5 * k * (r-r0)**2
        u[r > r0] = np.zeros( np.shape(u[r > r0]) )
        return u

class TabulatedNonbonded(AbstractPotential):
    def __init__(self, tableFile, *args, **kwargs):
        self.tableFile = tableFile
        AbstractPotential.__init__(self,*args,**kwargs)

        ## TODO: check that tableFile exists and is regular file

    def potential(self, r, types):
        raise NotImplementedError('This should probably not be implemented')
        
    def write_file(self, filename, types):
        if filename != self.tableFile:
            copyfile(self.tableFile, filename)

class BoundaryPotential(AbstractPotential):
    """Boundary potential for confining simulations."""
    
    def __init__(self, cell_vectors, cell_origin=None, well_depth=1.0, 
                 resolution=2.0, blur=5.0,filename="boundary.dx", **kwargs):
        """Initialize boundary potential.
        
        Args:
            cell_vectors: List of 3 cell basis vectors [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]]
            cell_origin: Cell origin coordinates [x,y,z]
            well_depth: Depth of potential well
            resolution: Grid resolution
            blur: Smoothing parameter
            filename: Output filename
            **kwargs: Additional parameters for AbstractPotential
        """
        default_range = (0, np.max([np.linalg.norm(v) for v in cell_vectors]))
        if 'range_' not in kwargs:
            kwargs['range_'] = default_range
        if 'resolution' not in kwargs:
            kwargs['resolution'] = resolution
            
        super().__init__(**kwargs)
        
        self.cell_vectors = cell_vectors
        self.cell_origin = cell_origin or [0, 0, 0]
        self.well_depth = well_depth
        self.blur = blur
        self.grid_cache = None
        self.filename=filename
    
    def _find_boundary_resolution(self):
        """Find appropriate boundary resolution for the cell vectors."""
        v1, v2, v3 = self.cell_vectors
        min_PBC_length = min([max(v1), max(v2), max(v3)])
        
        if min_PBC_length < self.resolution:
            dd = round(min_PBC_length / 2)
        else:
            dd = self.resolution

        n1 = round(np.linalg.norm(v1)/dd)
        n2 = round(np.linalg.norm(v2)/dd)
        n3 = round(np.linalg.norm(v3)/dd)

        return dd, n1, n2, n3
    
    def _find_boundary_end_points(self):
        """Find boundary endpoints from cell vectors and origin."""
        v1, v2, v3 = self.cell_vectors
        
        xyz_travel = [sum([v1[ind], v2[ind], v3[ind]]) for ind in range(3)]
        xyz_min = [self.cell_origin[ind] - xyz_travel[ind] * 0.5 for ind in range(3)]
        xyz_max = [self.cell_origin[ind] + xyz_travel[ind] * 0.5 for ind in range(3)]

        return (xyz_min[0], xyz_min[1], xyz_min[2], xyz_max[0], xyz_max[1], xyz_max[2])
    
    def _create_boundary_region(self, mx, my, mz, n1, n2, n3):
        """Create boundary region points for the cell."""
        v1, v2, v3 = self.cell_vectors
        v1_scaled = np.array(v1) / n1
        v2_scaled = np.array(v2) / n2
        v3_scaled = np.array(v3) / n3
        
        region_pts = []
        start_pt = np.array([mx, my, mz])
        
        for i in range(n1 + 1):
            for j in range(n2 + 1):
                for k in range(n3 + 1):
                    region_pts.append(start_pt + i * v1_scaled + j * v2_scaled + k * v3_scaled)

        return region_pts
    
    def _convert_to_mesh_points(self, region_pts, min_wallX, min_wallY, min_wallZ, dx, dy, dz, grid_shape):
        """Convert region points to grid indices."""
        mesh_pts = []
        for pt in region_pts:
            indx = round((pt[0] - min_wallX) / dx)
            indy = round((pt[1] - min_wallY) / dy)
            indz = round((pt[2] - min_wallZ) / dz)
            if (0 <= indx < grid_shape[0] and 0 <= indy < grid_shape[1] and 0 <= indz < grid_shape[2]):
                mesh_pts.append((indx, indy, indz))
        return mesh_pts
    
    def _create_rectangular_mesh(self, wall_range_x, wall_range_y, wall_range_z, dd):
        """Create a rectangular mesh grid."""
        x = np.linspace(wall_range_x[0], wall_range_x[1], int((wall_range_x[1]-wall_range_x[0])/dd) + 1)
        y = np.linspace(wall_range_y[0], wall_range_y[1], int((wall_range_y[1]-wall_range_y[0])/dd) + 1)
        z = np.linspace(wall_range_z[0], wall_range_z[1], int((wall_range_z[1]-wall_range_z[0])/dd) + 1)
        
        dx = np.mean(np.diff(x))
        dy = np.mean(np.diff(y))
        dz = np.mean(np.diff(z))

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        return X, Y, Z, dx, dy, dz
    
    def _blur_3d_grid(self, grid, blur_size, dd):
        """Apply 3D Gaussian blur to a grid."""
        # Create Gaussian kernel
        sideLen = 2*int(blur_size*3)+1
        gauss = scipy.signal.windows.gaussian(sideLen, blur_size/dd)
        i = np.arange(sideLen)
        i, j, k = np.meshgrid(i, i, i, indexing='ij')
        kernel = gauss[i]*gauss[j]*gauss[k]
        kernel = kernel/kernel.sum()
        
        # Apply convolution
        return scipy.signal.fftconvolve(grid, kernel, mode='same')
    
    def potential(self, r, types=None):
        """Calculate potential at positions.
        
        This isn't used directly for grid potentials in ARBD, but implemented 
        for consistency with the AbstractPotential interface.
        """
        # Calculate distance from boundary
        max_dist = np.max([np.linalg.norm(v) for v in self.cell_vectors]) / 2
        
        potential = np.zeros_like(r)
        for i, ri in enumerate(r):
            # Simplified: calculate distance from origin in normalized coordinates
            dist_vec = np.array([
                ri/np.linalg.norm(self.cell_vectors[0]),
                ri/np.linalg.norm(self.cell_vectors[1]),
                ri/np.linalg.norm(self.cell_vectors[2])
            ])
            
            # Distance from boundary (simplified)
            dist_from_boundary = max_dist - np.max(np.abs(dist_vec))

            if dist_from_boundary < 0:
                potential[i] = self.well_depth * (1 - np.exp(dist_from_boundary/self.blur))
                
        return potential
    
    def write_file(self, filename=None, types=None):
        """Write boundary potential to grid file."""
        
        # Find optimal resolution
        dd, n1, n2, n3 = self._find_boundary_resolution()
        
        # Find boundary endpoints
        mx, my, mz, Mx, My, Mz = self._find_boundary_end_points()
        
        # Set up wall ranges with buffer
        scale = 1.5  # Scale factor for wall range
        wall_range_x = (scale*(mx-2*self.blur), scale*(Mx+2*self.blur))
        wall_range_y = (scale*(my-2*self.blur), scale*(My+2*self.blur))
        wall_range_z = (scale*(mz-2*self.blur), scale*(Mz+2*self.blur))
        
        # Create mesh
        X, Y, Z, dx, dy, dz = self._create_rectangular_mesh(
            wall_range_x, wall_range_y, wall_range_z, dd)
        
        # Create boundary points
        region_pts = self._create_boundary_region(
            mx, my, mz, n1, n2, n3)
        
        # Convert to mesh points
        mesh_pts = self._convert_to_mesh_points(
            region_pts, wall_range_x[0], wall_range_y[0], wall_range_z[0], 
            dx, dy, dz, X.shape)
        
        # Create potential
        pot = np.zeros(np.shape(X))
        for pt in mesh_pts:
            pot[pt[0], pt[1], pt[2]] = -self.well_depth  # Negative for attraction
        
        # Apply Gaussian blur
        pot_blur = self._blur_3d_grid(pot, self.blur, dd)
        
        # Write to DX file
        origin_out = [wall_range_x[0], wall_range_y[0], wall_range_z[0]]
        delta = [dx, dy, dz]
        writeDx(self.filename, pot_blur, origin_out, delta)
        
        return self.filename

class NullPotential(AbstractPotential):
    def __init__(self, range_=(0,1), resolution=0.5, filename_prefix='./potentials/', *args, **kwargs):
        self.filename_prefix = filename_prefix
        AbstractPotential.__init__(self, range_=range_, resolution=resolution, *args,**kwargs)

    def potential(self, r, types):
        return np.zeros(r.shape)

    def filename(self, types=None):
        return f"{self.filename_prefix}nullpot.dat"

## Bonded potentials            
class HarmonicBondedPotential(AbstractPotential):
    def __init__(self, k, r0, filename_prefix='./potentials/', *args, **kwargs):
        self.k = k
        self.r0 = r0
        self.filename_prefix = filename_prefix
        if 'zero' not in kwargs: kwargs['zero'] = 'min'
        AbstractPotential.__init__(self, *args, **kwargs)

    @property
    def kscale(self):
        return 1.0
 
    @property
    @abstractmethod
    def type_(self):
        return "none"
   
    def filename(self, types=None):
        assert(types is None)
        return f"{self.filename_prefix}{self.type_}-{self.k*self.kscale:.3f}-{self.r0:.3f}.dat"
    
    def potential(self, r, types=None):
        dr = r-self.r0
        if self.periodic == True:
            r_span = self.range_[1]-self.range_[0]
            assert(r_span > 0)
            dr = np.mod( dr+0.5*r_span, r_span) - 0.5*r_span 
        return 0.5*self.k*dr**2

    def __hash__(self):
        assert(self.type_ != "None")
        return hash((self.type_, self.k, self.r0, self.filename_prefix, self.periodic, AbstractPotential.__hash__(self))) # 

    def __eq__(self, other):
        for a in ("type_", "k", "r0", "filename_prefix", "periodic"):
            if getattr(self,a) != getattr(other,a):
                return False
        return AbstractPotential.__eq__(self,other)
    
class HarmonicBond(HarmonicBondedPotential):
    def __init__(self, k, r0, correct_geometry=False, temperature=295, *args, **kwargs):
        if 'range_' not in kwargs: kwargs['range_'] = (0,50)
        HarmonicBondedPotential.__init__(self, k, r0, *args, **kwargs)
        self.correct_geometry = correct_geometry
        self.temperature = temperature

    @property
    def kscale(self):
        return 1.0
 
    @property
    def type_(self):
        return 'gbond' if self.correct_geometry else 'bond'

    def potential(self, r, types=None):
        u = HarmonicBondedPotential.potential(self, r, types)
        if self.correct_geometry:
            with np.errstate(divide='ignore',invalid='ignore'):
                du = 2*0.58622592*np.log(r) * self.temperature/295
            du[np.logical_not(np.isfinite(du))] = 0
            u = u+du
        return u

class HarmonicAngle(HarmonicBondedPotential):
    def __init__(self, *args, **kwargs):
        if 'range_' not in kwargs: kwargs['range_'] = (0,181)
        HarmonicBondedPotential.__init__(self, *args, **kwargs)

    @property
    def kscale(self):
        return (180.0/np.pi)**2

    @property
    def type_(self):
        return 'angle'

class HarmonicDihedral(HarmonicBondedPotential):
    def __init__(self, *args, **kwargs):
        if 'range_' not in kwargs: kwargs['range_'] = (-180,180)
        HarmonicBondedPotential.__init__(self, *args, **kwargs)

    @property
    def kscale(self):
        return (180.0/np.pi)**2

    @property
    def type_(self):
        return 'dihedral'

    @property
    def periodic(self):
        return True


class HarmonicVectorAngle(HarmonicBondedPotential):
    def __init__(self, *args, **kwargs):
        if 'range_' not in kwargs: kwargs['range_'] = (0,180)
        HarmonicBondedPotential.__init__(self, *args, **kwargs)

    @property
    def kscale(self):
        return (180.0/np.pi)**2

    @property
    def type_(self):
        return 'vecangle'


class WLCSKPotential(HarmonicBondedPotential, metaclass=ABCMeta):
    """ ## https://aip.scitation.org/doi/full/10.1063/1.4968020 """
    def __init__(self, d, lp, kT, **kwargs):
        ## Note, we're leveraging the HarmonicBondedPotential base class and set k to lp here, but it isn't proper
        HarmonicBondedPotential.__init__(self, d, lp, **kwargs)
        self.d = d          # separation
        self.k = self.lp = lp            # persistence length
        self.kT = kT

    def filename(self,types=None):
        assert(types is None)
        return "%s%s-%.3f-%.3f.dat" % (self.filename_prefix, self.type_,
                                       self.d, self.lp)

    def __hash__(self):
        try: self.k             # Preferably do not commit (needed to load some pkl model)
        except: self.k = self.lp
        try: self.filename_prefix ## also
        except: self.filename_prefix = self.prefix
        return hash((self.d, self.lp, self.kT, HarmonicBondedPotential.__hash__(self)))

    def __eq__(self, other):
        for a in ("d", "lp", "kT"):
            if self.__dict__[a] != other.__dict__[a]:
                return False
        return HarmonicBondedPotential.__eq__(self,other)
    
class WLCSKBond(WLCSKPotential):
    """ ## https://aip.scitation.org/doi/full/10.1063/1.4968020 """
    def __init__(self, d, lp, kT, **kwargs):
        if 'range_' not in kwargs: kwargs['range_'] = (0,50)
        if 'resolution' not in kwargs: kwargs['resolution'] = 0.02
        if 'max_force' not in kwargs: kwargs['max_force'] = 100
        WLCSKPotential.__init__(self, d, lp, kT, **kwargs)

    @property
    def type_(self):
        return "wlcbond"

    def potential(self, r, types=None):
        dr = r
        nk = self.d / (2*self.lp)
        q2 = (dr / self.d)**2
        a1,a2 = 1, -7.0/(2*nk)
        a3 = 3.0/32 - 3.0/(8*nk) - 6.0/(4*nk**2)
        p0,p1,p2,p3,p4 = 13.0/32, 3.4719,2.5064,-1.2906,0.6482
        a4 = (p0 + p1/(2*nk) + p2*(2*nk)**-2) / (1+p3/(2*nk)+p4*(2*nk)**-2)
        with np.errstate(divide='ignore',invalid='ignore'):
            u = self.kT * nk * ( a1/(1-q2) - a2*np.log(1-q2) + a3*q2 - 0.5*a4*q2*(q2-2) )
        return u

class WLCSKAngle(WLCSKPotential):
    """ ## https://aip.scitation.org/doi/full/10.1063/1.4968020 """
    def __init__(self, d, lp, kT, **kwargs):
        if 'range_' not in kwargs: kwargs['range_'] = (0,181)
        if 'resolution' not in kwargs: kwargs['resolution'] = 0.5
        WLCSKPotential.__init__(self, d, lp, kT, **kwargs)

    @property
    def type_(self):
        return "wlcangle"

    def potential(self, r, types=None):
        dr = r - 180
        nk = self.d / (2*self.lp)
        p1,p2,p3,p4 = -1.237, 0.8105, -1.0243, 0.4595
        C = (1 + p1*(2*nk) + p2*(2*nk)**2) / (2*nk+p3*(2*nk)**2+p4*(2*nk)**3)
        u = self.kT * C * (1-np.cos(dr * np.pi / 180))
        return u

