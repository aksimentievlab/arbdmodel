from __future__ import absolute_import, print_function
import numpy as np 
from scipy import signal
import os,sys
from . import logger
import unittest
import numpy as np



def add_smaller_grid( grid, smaller_grid ):
    slices = get_slice_enclosing_smaller_grid( grid, smaller_grid )
    grid.grid[slices] = grid.grid[slices] + smaller_grid.grid

def average_grids(grids, mask='nan'):
    """
    Compute the average of multiple grids.

    :param grids: The input grids to average.
    :type grids: list of ndarrays
    :param mask: The mask type to use for averaging. If 'nan', only non-NaN values are considered. Default is 'nan'.
    :type mask: str
    :return: The average grid.
    :rtype: ndarray
    :raises NotImplementedError: If the mask option is not implemented.

    Example:
        >>> grid1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> grid2 = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]])
        >>> averaged_grid = average_grids([grid1, grid2], mask='nan')
        >>> print(averaged_grid)
    """
    
    def __get_mask(grid, mask):
      if mask == 'nan':
        g_mask = ~np.isnan(grid)
      else:
        raise NotImplementedError
      return g_mask
      
    counts = np.zeros(grids[0].shape, dtype=int)
    totals = np.zeros(counts.shape)
    for g in grids:
      g_mask = __get_mask(g, mask)
      counts = counts + g_mask
      totals[g_mask] = totals[g_mask] + g[g_mask]
    sl = counts > 0

    average = totals
    average[sl] = totals[sl] / counts[sl]
    average[counts == 0] = np.nan
    return average

def loadGrid(file, **kwargs):
    """Load grid data from DX file"""
    # Read DX file header to get dimensions and metadata
    with open(file, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    for line in lines:
        if 'counts' in line:
            dims = [int(x) for x in line.split()[-3:]]
            break
            
    for line in lines:
        if 'origin' in line:
            origin = [float(x) for x in line.split()[-3:]]
            break
            
    for line in lines:
        if 'delta' in line:
            delta = float(line.split()[1])
            break
            
    # Read data values
    data = []
    for line in lines:
        try:
            values = [float(x) for x in line.split()]
            data.extend(values)
        except ValueError:
            continue
            
    # Reshape into 3D array
    grid = np.array(data).reshape(dims)
    return grid, origin, delta

def Bound_grid(inFile, outFile, lowerBound, upperBound):
    """Apply bounds to grid values"""
    # Fix scientific notation
    cmd_in = "sed -r 's/^([0-9]+)e/\1.0e/g; s/ ([0-9]+)e/ \1.0e/' " + inFile + " > bound_grid_temp0.dx"
    os.system(cmd_in)
    cmd_in = "sed -r 's/^(-[0-9]+)e/\1.0e/g; s/ (-[0-9]+)e/ \1.0e/' bound_grid_temp0.dx > bound_grid_temp1.dx"
    os.system(cmd_in)

    assert(lowerBound < upperBound)

    # Load data
    grid, origin, delta = loadGrid('bound_grid_temp1.dx')

    # Apply bounds
    grid[grid > upperBound] = upperBound
    grid[grid < lowerBound] = lowerBound

    # Write output
    writeDx(outFile, grid, origin, [delta, delta, delta])

def blur3Dgrid(g, blur):
    """Apply 3D Gaussian blur"""
    kernel = gaussian_kernel(voxels=2*int(blur*3)+1, sig=blur, ndim=3)
    return signal.fftconvolve(g, kernel, mode='same')

    
def Create_a_rectangular_mesh(wallRangeX, wallRangeY, wallRangeZ, blur, dd):
    x = np.linspace(wallRangeX[0], wallRangeX[1], int((wallRangeX[1]-wallRangeX[0])/dd) + 1)
    y = np.linspace(wallRangeY[0], wallRangeY[1], int((wallRangeY[1]-wallRangeY[0])/dd) + 1)
    z = np.linspace(wallRangeZ[0], wallRangeZ[1], int((wallRangeZ[1]-wallRangeZ[0])/dd) + 1)
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))
    dz = np.mean(np.diff(z))

    X,Y,Z = np.meshgrid(x,y,z,indexing='ij')

    return X, Y, Z, dx, dy, dz



def Convert_math_pt_to_mesh_pt(region_pts, min_wallX, min_wallY, min_wallZ, dx, dy, dz):
    mesh_pts = []
    for pt in region_pts:
        indx = round((pt[0] - min_wallX) / dx)
        indy = round((pt[1] - min_wallY) / dy)
        indz = round((pt[2] - min_wallZ) / dz)
        mesh_pts.append((indx, indy, indz))
    return mesh_pts

def Create_the_well(X, wellDepth, mesh_pts, blur, dx, dy, dz, wX, wY, wZ, out_path):
    """Create potential well"""
    pot = np.zeros(np.shape(X))
    for pt in mesh_pts:
        pot[pt[0], pt[1], pt[2]] = wellDepth

    dd = np.mean([dx, dy, dz])
    pot_blur = blur3Dgrid(pot, blur/dd)

    origin = [wX, wY, wZ]
    writeDx(out_path, pot_blur, origin, [dd, dd, dd])

def Create_null(grid_path='null.dx'):
    """Create null potential grid"""
    zeros = np.zeros([2,2,2])
    origin = -1500*np.array((1,1,1))
    delta = [3000, 3000, 3000]
    writeDx(grid_path, zeros, origin, delta)

class TestAverageGrids(unittest.TestCase):
    def test_average_grids(self):
        grid1 = np.array([[1, 2, np.nan], [4, 5, 6], [7, 8, 9]])
        grid2 = np.array([[2, 4, 6], [8, np.nan, 12], [14, 16, 18]])
        expected_result = np.array([[1.5, 3, 6], [6, 5, 9], [10.5, 12, 13.5]])
        
        result = average_grids([grid1, grid2], mask='nan')
        
        np.testing.assert_array_equal(result, expected_result)

        
def neighborhood_average(grid, neighborhood = 1, fill_value='mirror'):
    """
    Compute the neighborhood average of a grid.

    Parameters:
        grid (ndarray): The input grid.
        neighborhood (int or list): The size of the neighborhood to use for averaging. If an integer is provided, the same neighborhood size is used for all dimensions. If a list is provided, it should specify the neighborhood size for each dimension separately. Default is 1.
        fill_value (str or ndarray): The fill value to use for padding the grid. If 'mirror', values are mirrored along each dimension. If 'nearest', values are filled with the nearest neighbor. If 'periodic', values are filled with periodic boundary conditions. If a number is provided, it is used directly as the fill value. Default is 'mirror'.

    Returns:
        ndarray: The grid with the neighborhood average computed.

    Notes:
        - Currently, only the 'mirror' fill value option is implemented.
        - This function requires the 'average_grids' function from an external source.

    Example:
        >>> grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> averaged_grid = neighborhood_average(grid, neighborhood=2, fill_value='mirror')
        >>> print(averaged_grid)
    """
    try:
        neighborhood[0]
    except:
        neighborhood = [neighborhood] * grid.ndim

    padded_grid = np.ones( [s+dx*2 for s,dx in zip(grid.shape,neighborhood)] )*np.nan
    if fill_value == 'mirror':
      def get_slices(mirror_dims, left):
        sl1 = tuple(slice(n,-n) if i not in mirror_dims else 
                    slice(None,n) if left[mirror_dims.index(i)]
                    else slice(-n,None)
                    for i,n in enumerate(neighborhood))
        sl2 = tuple(slice(None,None) if i not in mirror_dims else
                    slice((n-1),None,-1) if left[mirror_dims.index(i)]
                    else slice(None,-(n+1),-1)
                    for i,n in enumerate(neighborhood))
        return sl1,sl2

      def build_dim_arrays(nd, mirror_dims, left):
        d0 = mirror_dims[-1] if len(mirror_dims) > 0 else -1
        mirror_dims.append(d0)
        left.append(False)
        for d in range(d0+1,grid.ndim+1-nd):
          mirror_dims[-1] = d
          left[-1] = False
          yield list(mirror_dims), list(left)
          left[-1] = True
          yield list(mirror_dims), list(left)
                
      all_mirror_dims_arrays = []
      all_left_arrays = []
      for nd in range(1,grid.ndim+1):
        # print(nd,grid.ndim)
        mirror_dims_arrays = [[]]
        left_arrays = [[]]
        while nd > 0:
          for m0,l0 in list(zip(mirror_dims_arrays,left_arrays)):
            for m,l in build_dim_arrays(nd, m0, l0):
              mirror_dims_arrays.append(m)
              left_arrays.append(l)
          nd=nd-1
        # print("dbg",nd)  
        # print(mirror_dims_arrays)
        all_mirror_dims_arrays.extend(mirror_dims_arrays)
        all_left_arrays.extend(left_arrays)

      for m,l in zip(all_mirror_dims_arrays, all_left_arrays):
        ## set values mirroring d
        sl1,sl2 = get_slices(m, l)
        padded_grid[sl1] = grid[sl2]
    elif fill_value == 'nearest':
      raise NotImplementedError      
    elif fill_value == 'periodic':
      raise NotImplementedError
    else:
      padded_grid = fill_value   

    sl = tuple(slice(n,-n) for n in neighborhood)
    padded_grid[sl] = grid
    # print(padded_grid)

    ## Gather sliced arrays to perform average
    def get_slice(neighborhood, slices=None):
      # print("get_slice",neighborhood)
      if slices is None: slices = [[]]
      if len(neighborhood) > 1:
        slices = get_slice(neighborhood[1:], slices)
      nmax = 2*neighborhood[0]
      slices = [[slice(n,-(nmax-n) if nmax != n else None)]+sls for n in range(nmax+1) for sls in slices]
      return slices

    slices = get_slice(neighborhood)
    result = average_grids([padded_grid[sl] for sl in slices])
    return result
  
def replace_false_with_distance(boolean_grid, sampling):
    import ndimage
    return ndimage.morphology.distance_transform_edt( boolean_grid, sampling=[Dxy,Dxy,Dz] )

  
def fill_nans(grid, neighborhood=1, max_iterations=np.inf, mask=None):
    """
    Fill NaN values in a grid using neighborhood averaging.

    Parameters:
        grid (ndarray): The input grid containing NaN values.
        neighborhood (int, optional): The size of the neighborhood to use for averaging. Default is 1.
        max_iterations (int, optional): The maximum number of iterations to perform. Default is infinity.
        mask (ndarray, optional): A mask specifying which values to consider for filling NaNs. If None, all non-NaN values are used. Default is None.

    Returns:
        ndarray: The grid with NaN values filled using neighborhood averaging.

    Notes:
        - This function requires the 'skimage' package, specifically the 'find_boundaries' function from 'skimage.segmentation'.

    Example:
        >>> grid = np.array([[1, 2, np.nan], [4, np.nan, 6], [np.nan, 8, 9]])
        >>> filled_grid = fill_nans(grid, neighborhood=2, max_iterations=10)
        >>> print(filled_grid)
    """
    from skimage.segmentation import find_boundaries

    ret = np.array(grid)
    if mask is None: mask = ~np.isnan(grid)
    assert(np.all(np.isnan(mask)) == 0)
    nans= np.isnan(ret)

    i = 0
    while np.sum(nans) > 0:
      if i > max_iterations: break
      i+=1
      print("nans",np.sum(nans))
      boundary=find_boundaries(~nans, mode='outer')
      tmp = neighborhood_average(ret, neighborhood)
      ret[boundary] = tmp[boundary]
      ret[mask] = grid[mask]
      nans = np.isnan(ret)
    return ret

def convolve_kernel_truncate( array, kernel ):
    """
    Convolve an array with a kernel, truncating the output.

    Parameters:
        array (ndarray): The input array.
        kernel (ndarray): The kernel to convolve with the array.

    Returns:
        ndarray: The truncated convolution result.

    Raises:
        AssertionError: If the dimensions of the array and kernel do not match.
        AssertionError: If any dimension of the array is smaller than the corresponding dimension of the kernel.

    Notes:
        - This function assumes that the kernel has odd dimensions along each dimension.
        - If any dimension of the kernel has an even number of elements, a warning message is printed to stderr indicating that the output may be shifted.

    Example:
        >>> array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> kernel = np.array([[0.5, 0.5], [0.5, 0.5]])
        >>> result = convolve_kernel_truncate(array, kernel)
        >>> print(result)
    """
    arrayShape = np.shape( array )
    kernelShape = np.shape( kernel )
    assert( len(arrayShape) == len(kernelShape) )

    for an,kn in zip(arrayShape,kernelShape): assert(an > kn)

    dim = 0
    for an in kernelShape:
        dim += 1
        if an % 2 == 0: 
            print("WARNING: kernel has even number of elements along dimension %d; Output may be shifted." % dim, file=sys.stderr)


    
    initShape = [a+b for a,b in zip(arrayShape,kernelShape)]
    convolution = np.zeros(initShape)
    count = np.zeros(initShape)
    
    for idx in range( np.prod(kernelShape) ):
        
        ijk = [ int(idx/a) % b for a,b in
                  zip( np.hstack(([1],np.cumprod(kernelShape[:-1]))), kernelShape) ][::-1]

        s = tuple([ slice( (kn-i), an+kn-i ) for i,kn,an in 
                    zip(ijk,kernelShape,arrayShape) ])
        
        val = kernel.flatten()[idx]

        convolution[s] += val*array
        count[s] += 1
        
    ids = count > 0
    convolution[ids] = (convolution[ids]/count[ids]) * np.prod(kernelShape)
    

    s = tuple([ slice( int((kn-1)/2)+1, -int((kn-1)/2) ) for kn in kernelShape ])
    return convolution[s]

def isotropic_kernel( function, delta, shape = None, cutoff = None, normalize = False):
    """
    Generate an isotropic kernel function.

    Parameters:
        function (callable): A callable representing the kernel function.
            It should accept a scalar or an array-like object and return a scalar or an array-like object of the same shape.
        delta (list or tuple): A list or tuple specifying the spacing between grid points in each dimension.
        shape (list or tuple, optional): A list or tuple specifying the shape of the kernel. If None, it is determined based on the cutoff parameter.
        cutoff (scalar, optional): A scalar specifying the maximum distance from the center to consider for the kernel. If provided, it determines the shape of the kernel based on the delta values.
        normalize (bool, optional): A boolean indicating whether to normalize the kernel. If True, the kernel values are divided by the sum of all kernel values, ensuring that the kernel sums to 1.

    Returns:
        ndarray: An ndarray representing the isotropic kernel.

    Raises:
        ValueError: If the cutoff parameter is None, as it is required to determine the shape of the kernel.

    Example:
        >>> import numpy as np
        >>> def gaussian(x):
        ...     return np.exp(-0.5 * x**2)
        >>> delta = [0.1, 0.1, 0.1]
        >>> kernel = isotropic_kernel(gaussian, delta, cutoff=1.0, normalize=True)
        >>> print(kernel)

    """

    if cutoff is not None:
        shape = [(cutoff*2)//d for d in delta]
    elif cutoff is None:
        raise ValueError

    ndim = len(shape)
    assert( len(delta) == ndim )

    centers = [(np.arange(n)-(n-1)*0.5)*d for n,d in zip(shape,delta)]
    CENTERS = np.meshgrid(*centers, indexing='ij')
    R = np.sqrt( sum([C**2 for C in CENTERS]) )
    kernel = function(R)
    if normalize:
        kernel = kernel/np.sum(kernel)
    return kernel

def gaussian_kernel(voxels=5, sig=1., ndim=3):
    ## TODO: rewrite using isotropic_kernel
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    gauss_list = []
    try:
        sig[0]
    except:
        sig = [sig]*ndim
    try:
        voxels[0]
    except:
        voxels = [voxels]*ndim
    
    for l,s in zip(voxels,sig):
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        gauss_list.append( np.exp(-0.5 * np.square(ax) / np.square(s)) )

    gauss_list2 = []
    kernel = gauss_list[0] # np.empty((0))
    for i,g in enumerate(gauss_list[1:]):
        i=i+1
        sl = tuple(slice(None) if j==i else None for j in range(i))
        kernel = kernel[...,None]*g[sl]
    
    return kernel / np.sum(kernel)


def slab_potential_z(force_constant, center, dimensions, resolution, exponent=2):
  try:
    resolution[0]
  except:
    resolution = 3*[resolution]
    
  n_voxels = [int(np.ceil(w/r)) for w,r in zip(dimensions,resolution)]
  x,y,z = [np.arange(n)*r-(c+(n/2.0)*r) for n,r,c in zip(n_voxels, resolution, center)]
  u = (force_constant/exponent) * np.abs((z-center[0])**exponent) # 1D array
  U = np.empty( n_voxels )
  U = u[None,None,:]  
  return U

def constant_force(force, dimensions, resolution, origin=None):
  try:
    force[0]
  except:
    force = [0,0,force]
    
  try:
    resolution[0]
  except:
    resolution = 3*[resolution]

  if origin is None:
      origin = [-0.5 * _x for _x in dimensions]

  n_voxels = [int(np.ceil(w/r)) for w,r in zip(dimensions,resolution)]
  x,y,z = [(np.arange(n)+0.5)*r+o for n,r,o in zip(n_voxels, resolution, origin)]
  X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
  
  U = force[0]*X + force[1]*Y + force[2] * Z
  return U

def spherical_confinement(force_constant, radius, dimensions,  resolution, center=(0,0,0), exponent=2):
  try:
    force[0]
  except:
    force = [0,0,force]
    
  try:
    resolution[0]
  except:
    resolution = 3*[resolution]
    
  n_voxels = [int(np.ceil(w/r)) for w,r in zip(dimensions,resolution)]
  x,y,z = [np.arange(n)*r-(c+(n/2.0)*r) for n,r,c in zip(n_voxels, resolution, center)]
  X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
  R = np.sqrt(X**2+Y**2+Z**2)
  U = np.empty(R.shape)
  sl = R > radius
  U[sl] = (force_constant/exponent) * (R-radius)**exponent
  U[~sl] = 0
  return U

  
def writeDx(outfile, data, origin, delta, fmt="%.12f"):
  shape = np.shape(data)
  num = np.prod(shape)
  assert( len(shape) == 3 )
  assert( len(origin) == 3 )
  assert( len(delta) == 3 )
  headerInfo = dict( nx=shape[0], ny=shape[1], nz=shape[2],
                     ox=origin[0], oy=origin[1], oz=origin[2],
                     dx=delta[0], dy=delta[1], dz=delta[2],
                     num=num
                   )
  data = data.flatten(order='C')
  header = """# OpenDX density file
# File format: http://opendx.sdsc.edu/docs/html/pages/usrgu068.htm#HDREDF
object 1 class gridpositions counts  {nx} {ny} {nz}
origin {ox} {oy} {oz}
delta  {dx} 0.000000 0.000000
delta  0.000000 {dy} 0.000000
delta  0.000000 0.000000 {dz}
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



## Utility functions

def create_bounding_grid( *grids ):
    """ Construct a grid bigger than all the (GridDataFormats) grids provided in the arguments; note that the inputs must be orthonormal and have exactly overlapping voxels """
    def _create_bounding_grid( grid1, grid2 ):
      assert( len(grid1.grid.shape) == 3 )
      assert( len(grid2.grid.shape) == 3 )
      assert(len(grid1.delta) == 3)
      assert( np.all(grid1.delta == grid2.delta) )
      min_origin = np.minimum(*[g.origin for g in (grid1, grid2)])
      max_shape = np.maximum(*[(g.origin-min_origin)//g.delta + np.array(g.grid.shape) for g in (grid1, grid2)]).astype(int)
      assert( np.all( max_shape > 0) )
      
      new_grid_data = np.empty(max_shape)
      new_grid = Grid( new_grid_data, origin = min_origin, delta = grid1.delta )
      return new_grid

    g1 = grids[0]
    if len( grids ) == 1:
      g1 = _create_bounding_grid( g1, g1 )
    else:
      for g2 in grids[1:]:
        g1 = _create_bounding_grid( g1, g2 )
    g1.grid[...,:] = 0

    return g1

def get_slice_enclosing_smaller_grid( grid, smaller_grid ):
    
    ## Check grids are orthonormal
    assert( len(grid.grid.shape) == 3 )
    assert( len(smaller_grid.grid.shape) == 3 )
    assert(len(grid.delta) == 3)

    ## Make sure grid spacing is the same
    assert( np.all(grid.delta == smaller_grid.delta) )
    delta = smaller_grid.delta

    ## Find the offset between the grids and make sure smaller_grid is inside grid
    offset = (smaller_grid.origin - grid.origin)
    assert( np.all( offset > 0 ) )

    ## Check that grids are overlapping exactly and find index offset
    slices = []
    for x,d,n,ln in zip(offset, delta, smaller_grid.grid.shape, grid.grid.shape):
        assert( x/d == x//d )
        i0 = int(round(x/d))
        assert(i0+n <= ln) # make sure that smaller_grid isn't too big for grid
        slices.append(slice(i0,i0+n))

    return slices


if __name__ == '__main__':
    unittest.main()
