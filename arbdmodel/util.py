""" Useful utility functions """

from scipy.ndimage import convolve
from scipy.signal import savgol_filter as svgf
import numpy as np


def savgol_coeffs_nd(window_length, polyorder, deriv=0, delta=1.0, pos=None,
                     use='conv', maxorder=None):
    """ Compute the coefficients for an N-D Savitzky-Golay FIR filter.

    Parameters
    ----------
    window_length : int
        The length of the filter window (i.e., the number of coefficients).
    polyorder : int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.
    deriv : int, optional
        The order of the derivative to compute. This must be a
        nonnegative integer. The default is 0, which means to filter
        the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0.
    pos : int or None, optional
        If pos is not None, it specifies evaluation position within the
        window. The default is the middle of the window.
    use : str, optional
        Either 'conv' or 'dot'. This argument chooses the order of the
        coefficients. The default is 'conv', which means that the
        coefficients are ordered to be used in a convolution. With
        use='dot', the order is reversed, so the filter is applied by
        dotting the coefficients with the data set.

    Returns
    -------
    coeffs : N-D ndarray
        The filter coefficients.

    Notes
    ------
    Roughly follows the approach described here:
    https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter#Two-dimensional_convolution_coefficients
    """

    if pos is not None:
        raise NotImplementedError
    if use != 'conv':
        raise NotImplementedError        

    if deriv != 0:
        raise NotImplementedError('Derivatives not yet supported')
    
    if maxorder is None:
        maxorder = np.max(polyorder)
    ndim = len(window_length)
    axes = [np.arange(v) - (v-1)//2 for v in window_length]
    J0 = [np.vander(a,o+1)[:,::-1] for a,o in zip(axes,polyorder)]

    ## Build combined vandermonde-like matrix
    J = []
    for o in np.arange(np.max(polyorder)+1):
        # print(f"Working on order {o}")
        ids = np.zeros(ndim,dtype=int)
        while ids[0] < o+1:
            ids[-1] = o - np.sum(ids[:-1])

            if ids.sum() <= maxorder and all(_i <= _o for _i,_o in zip(ids,polyorder)):
                Js = [_J0[:,i].flatten()[tuple(slice(None) if slidx == idx else None for slidx in range(ndim))]
                             for idx,(_J0,i) in enumerate(zip(J0,ids))]
                candidate = Js[0]
                for _J in Js[1:]:
                    candidate = candidate * _J
                J.append( candidate.flatten() )

            d = ndim-2
            while d >= 0:
                if ids[d] > polyorder[d]:
                    ids[d] = 0
                    d -= 1
                else:
                    ids[d] = ids[d] + 1
                    break

    J = np.array(J).T
    C = np.linalg.inv( J.T @ J) @ J.T

    ## Rearrange coefficients
    IJK = np.meshgrid(*[np.arange(w) for w in window_length], indexing='ij')
    kernel = np.zeros(IJK[0].shape)
    for c,*ijk in zip(C[0],*[A.flatten() for A in IJK]):
        kernel[tuple(slice(i,i+1) for i in ijk)] = c
    return kernel
    
def savgol_filter_nd(x, window_length, polyorder, deriv=0, delta=1.0,
                     mode = 'nearest', cval=0.0, maxorder=None):
    """Apply a Savitzky-Golay filter to an n-dimensional array.

    Parameters
    ----------
    x : array_like
        The data to be filtered. If `x` is not a single or double precision
        floating point array, it will be converted to type ``numpy.float64``
        before filtering.
    window_length : int or array_like of int with dimensions matching `x`
        The length of the filter window (i.e., the number of coefficients).
        If `mode` is 'interp', `window_length` must be less than or equal
        to the size of `x`.
    polyorder : int or array_like of int with dimensions matching `x`
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.
    deriv : int, optional
        The order of the derivative to compute. This must be a
        nonnegative integer. The default is 0, which means to filter
        the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0. Default is 1.0.
    mode : str, optional
        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'. This
        determines the type of extension to use for the padded signal to
        which the filter is applied.  When `mode` is 'constant', the padding
        value is given by `cval`.  See the Notes for more details on 'mirror',
        'constant', 'wrap', and 'nearest'.
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree `polyorder` polynomial is fit to the
        last `window_length` values of the edges, and this polynomial is
        used to evaluate the last `window_length // 2` output values. 
        Default is 'interp'.
    cval : scalar, optional
        Value to fill past the edges of the input if `mode` is 'constant'.
        Default is 0.0.
    maxorder : int, optional
        Largest order allowed for polynomial terms. Default is the
        largest value in `polyorder`.

    Returns
    -------
    y : ndarray, same shape as `x`
        The filtered data.

    Notes
    -----
    Machine precision may result in significant errors in the
    calculation of coefficients and subsequent convolution; the function
    has only been tested for short 3D windows, and your mileage may vary.

    Details on the `mode` options:

        'mirror':
            Repeats the values at the edges in reverse order. The value
            closest to the edge is not included.
        'nearest':
            The extension contains the nearest input value.
        'constant':
            The extension contains the value given by the `cval` argument.
        'wrap':
            The extension contains the values from the other end of the array.

        Other values will be passed to np.pad and may or may not work

    """

    x = np.asarray(x)

    ## Ensure that x is either single or double precision floating point.
    if x.dtype != np.float64 and x.dtype != np.float32:
        x = x.astype(np.float64)
        
    ndim = len(x.shape)

    if isinstance(window_length,int) or  isinstance(window_length,float):
        window_length = [window_length]*ndim

    if isinstance(polyorder,int) or isinstance(polyorder,float):
        polyorder = [polyorder]*ndim
        
    if any(a < 1 or a != (int(a-1)//2)*2+1 for a in window_length):
        raise ValueError(f'Window length {window_length} contains non-odd or negative entries')
    if any(a < 0 or a != int(a) for a in polyorder):
        raise ValueError(f'Ploynomial order {polyorder} contains non-integer or negative entries')

    if len(window_length) != ndim:
        raise ValueError(f'Window length {window_length} does not have the same dimensionality as input data array')
    if len(polyorder) != ndim:
        raise ValueError(f'Polynomial order {polyorder} does not have the same dimensionality as input data array')

    coeffs = savgol_coeffs_nd(window_length, polyorder, deriv=deriv, delta=delta, maxorder=maxorder)


    if mode == 'interp':
        if any(w > s for s,w in zip(x.shape, window_length)):
            raise ValueError(f"If mode is 'interp', window_length must be less "
                             "than or equal to the size of x along each axis.")
        # y = convolve(x, coeffs, mode="constant")
        # _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, axis, y)
        raise NotImplementedError
    else:
        y = convolve( x, coeffs, mode=mode, cval=cval )
    return y

def __test_savgol():
    inp = np.arange(60).reshape(10,6)
    s1 = savgol_filter_nd(inp,
                          window_length = [5,1],
                          polyorder = [3,0],
                          mode='nearest'
                          )

    s2 = [svgf(inp[:,i], window_length=5, polyorder=3, mode='nearest') for i in range(inp.shape[1])]

    for i,a1,a2 in zip(inp.T,s1.T,s2):
        print(i)
        print(a1)
        print(a2)

if __name__ == '__main__':
    __test_savgol()

