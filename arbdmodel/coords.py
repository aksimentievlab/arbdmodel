from .logger import logger
import numpy as np

def minimizeRmsd(coordsB, coordsA, weights=None, maxIter=100):
    ## Going through many iterations wasn't really needed
    tol = 1
    count = 0

    R = np.eye(3)
    comB = np.zeros([3,])
    cNext = coordsB

    while tol > 1e-6:
        q,cB,comA = _minimizeRmsd(cNext,coordsA, weights)
        R = R.dot(quaternion_to_matrix(q))
        assert( np.all(np.isreal( R )) )

        comB += cB
        cLast = cNext
        cNext = (coordsB-comB).dot(R)

        tol = np.sum(((cNext-cLast)**2)[:]) / np.max(np.shape(coordsB))
        if count > maxIter:
            Exception("Exceeded maxIter (%d)" % maxIter)
        count += 1

    print("%d iterations",count)
    return R, comB, comA


def minimizeRmsd(coordsB, coordsA, weights=None):
    q,comB,comA = _minimizeRmsd(coordsB, coordsA, weights)
    assert( np.all(np.isreal( q )) )
    return quaternion_to_matrix(q),comB,comA


## http://onlinelibrary.wiley.com/doi/10.1002/jcc.21439/full
def _minimizeRmsd(coordsB, coordsA, weights=None):
    A = coordsA
    B = coordsB

    shapeA,shapeB = [np.shape(X) for X in (A,B)]
    for s in (shapeA,shapeB):  assert( len(s) == 2 )

    A,B = [X.T if s[1] > s[0] else X for X,s in zip([A,B],(shapeA,shapeB))] # TODO: print warning

    shapeA,shapeB = [np.shape(X) for X in (A,B)]
    assert( shapeA == shapeB )
    for X,s in zip((A,B),(shapeA,shapeB)):
        assert( s[1] == 3 and s[0] >= s[1] )
    
    # if weights is None: weights = np.ones(len(A))
    if weights is None:
        comA,comB = [np.mean( X, axis=0 ) for X in (A,B)]
    else:
        assert( len(weights[:]) == len(B) )
        W = np.diag(weights)
        comA,comB = [np.sum( W.dot(X), axis=0 ) / np.sum(W) for X in (A,B)]

    A = np.array( A-comA )
    B = np.array( B-comB )

    if weights is None:
        s = A.T.dot(B)
    else:
        s = A.T.dot(W.dot(B))
    
    sxx,sxy,sxz = s[0,:]
    syx,syy,syz = s[1,:]
    szx,szy,szz = s[2,:]
    
    K = [[ sxx+syy+szz, syz-szy, szx-sxz, sxy-syx],
         [syz-szy,  sxx-syy-szz, sxy+syx, sxz+szx],
         [szx-sxz, sxy+syx, -sxx+syy-szz, syz+szy],
         [sxy-syx, sxz+szx, syz+szy, -sxx-syy+szz]]
    K = np.array(K)

    # GA = np.trace( A.T.dot(W.dot(A)) )
    # GB = np.trace( B.T.dot(W.dot(B)) )
        
    ## Finding GA/GB can be done more quickly
    # I = np.eye(4)
    # x0 = (GA+GB)*0.5
    # vals = newtoon(lambda x: np.det(K-x*I), x0 = x0)

    vals, vecs = np.linalg.eig(K)
    i = np.argmax(vals)
    q = vecs[:,i]

    # RMSD = np.sqrt( (GA+GB-2*vals[i]) / len(A) )
    # print("CHECK:", K.dot(q)-vals[i]*q )
    return q, comB, comA

def quaternion_to_matrix(q):
    """
    Convert a quaternion to a rotation matrix.
    
    This function converts a quaternion representation of a rotation into a 3x3 
    rotation matrix. The quaternion is automatically normalized before conversion.
    
    Parameters
    ----------
    q : array_like
        A quaternion represented as a 4-element array [q0, q1, q2, q3], where q0 
        is the scalar (real) part and q1, q2, q3 are the vector (imaginary) parts.
        
    Returns
    -------
    numpy.ndarray
        A 3x3 rotation matrix corresponding to the input quaternion.
        
    Notes
    -----
    The quaternion convention used here is (q0, q1, q2, q3) where q0 is the scalar
    part. The quaternion is normalized internally to ensure it represents a valid
    rotation.
    
    The Wikipedia article referenced employed a less common convention for quaternions:
    http://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rotation_matrix_.E2.86.94_quaternion
    Alternative notation would be:
    q1,q2,q3,q4 = q
    R = [[1-2*(q2*q2 + q3*q3),    2*(q1*q2 - q3*q4),    2*(q1*q3 + q2*q4)],
         [  2*(q1*q2 + q3*q4),  1-2*(q1*q1 + q3*q3),    2*(q2*q3 - q1*q4)],
         [  2*(q1*q3 - q2*q4),    2*(q1*q4 + q2*q3),  1-2*(q2*q2 + q1*q1)]]
    
    Examples
    --------
    >>> import numpy as np
    >>> q = [1, 0, 0, 0]  # Identity quaternion
    >>> R = quaternion_to_matrix(q)
    >>> print(R)
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    """
    assert(len(q) == 4)

    q = q / np.linalg.norm(q)
    q0,q1,q2,q3 = q
    R = [[1-2*(q2*q2 + q3*q3),    2*(q1*q2 - q3*q0),    2*(q1*q3 + q2*q0)],
         [  2*(q1*q2 + q3*q0),  1-2*(q1*q1 + q3*q3),    2*(q2*q3 - q1*q0)],
         [  2*(q1*q3 - q2*q0),    2*(q1*q0 + q2*q3),  1-2*(q2*q2 + q1*q1)]]

    return np.array(R)

def quaternion_from_matrix(R):
    """
    Convert N rotation matrices (N,3,3) to quaternions (N,4) in
    (w,x,y,z) order using Vectorized SciPy/Eigen-style max-diagonal
    algorithm.
    """
    R = np.asarray(R)
    dim = len(R.shape)
    if dim == 2: R = R.reshape(1,*R.shape)

    ## We use an unusual convention for R, and rather than change
    ## the code everywhere we can simply take the transpose here
    R = np.transpose(R,(0,2,1))

    N = R.shape[0]
    r00 = R[:,0,0]; r01 = R[:,0,1]; r02 = R[:,0,2]
    r10 = R[:,1,0]; r11 = R[:,1,1]; r12 = R[:,1,2]
    r20 = R[:,2,0]; r21 = R[:,2,1]; r22 = R[:,2,2]

    K = np.empty((N,4,4))

    K[:,0,0] = r00 - r11 - r22
    K[:,0,1] = r10 + r01
    K[:,0,2] = r20 + r02
    K[:,0,3] = r12 - r21

    K[:,1,0] = K[:,0,1]
    K[:,1,1] = -r00 + r11 - r22
    K[:,1,2] = r21 + r12
    K[:,1,3] = r20 - r02

    K[:,2,0] = K[:,0,2]
    K[:,2,1] = K[:,1,2]
    K[:,2,2] = -r00 - r11 + r22
    K[:,2,3] = r01 - r10

    K[:,3,0] = K[:,0,3]
    K[:,3,1] = K[:,1,3]
    K[:,3,2] = K[:,2,3]
    K[:,3,3] = r00 + r11 + r22

    K /= 3.0

    ## Batched symmetric eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(K)

    ## Largest eigenvalue eigenvector
    idx = np.argmax(eigvals, axis=1)
    q = eigvecs[np.arange(N), :, idx]
    q = np.roll(q,1,axis=1)            # use (w,x,y,z) convention

    if dim == 2: q = q[0]
    return q

def __quaternion_from_matrix__deprecated( R ):
    e1 = R[0]
    e2 = R[1]
    e3 = R[2]
    
    # d1 = 0.5 * np.sqrt( 1+R[0,0]+R[1,1]+R[2,2] )
    # d2 = 0.5 * np.sqrt( 1+R[0,0]-R[1,1]-R[2,2] )
    # d2 = 0.5 * np.sqrt( 1+R[0,0]-R[1,1]-R[2,2] )
    # d2 = 0.5 * np.sqrt( 1+R[0,0]-R[1,1]-R[2,2] )

    d1 = 1+R[0,0]+R[1,1]+R[2,2]
    d2 = 1+R[0,0]-R[1,1]-R[2,2]
    d3 = 1-R[0,0]+R[1,1]-R[2,2]
    d4 = 1-R[0,0]-R[1,1]+R[2,2]
    
    maxD = max((d1,d2,d3,d4))
    d = 0.5 / np.sqrt(maxD)

    if d1 == maxD:
        return np.array(( 1.0/(4*d),
                          d * (R[2,1]-R[1,2]),
                          d * (R[0,2]-R[2,0]),
                          d * (R[1,0]-R[0,1]) ))
    elif d2 == maxD:
        return np.array(( d * (R[2,1]-R[1,2]),
                          1.0/(4*d),
                          d * (R[0,1]+R[1,0]),
                          d * (R[0,2]+R[2,0]) ))
    elif d3 == maxD:
        return np.array(( d * (R[0,2]-R[2,0]),
                          d * (R[0,1]+R[1,0]),
                          1.0/(4*d),
                          d * (R[1,2]+R[2,1]) ))
    elif d4 == maxD:
        return np.array(( d * (R[1,0]-R[0,1]),
                          d * (R[0,2]+R[2,0]),
                          d * (R[1,2]+R[2,1]),
                          1.0/(4*d) ))

def quaternion_product(a, b):
    assert(len(a) == 4)
    assert(len(b) == 4)
    ab = np.empty(4)
    ab[0] = a[0]*b[0]-a[1:].dot(b[1:])
    ab[1:] = a[0]*b[1:] + b[0]*a[1:] + np.cross(a[1:],b[1:])
    return ab

def quaternion_inverse(q):
    assert(len(q) == 4)
    qinv = np.array(q)
    qinv[1:] = -qinv[1:]
    qinv = qinv / (q[0]**2 + q[1:].dot(q[1:]))
    return qinv

def quaternion_exp(q,t):
    assert(len(q) == 4)
    omega = np.arccos(q[0])
    v = q[1:]
    v = v/np.linalg.norm(v)
    qexp = np.empty(4)
    qexp[0] = np.cos(omega*t)
    qexp[1:] = np.sin(omega*t)*v
    return qexp

def quaternion_slerp(q1,q2,t):
    assert(len(q1) == 4)
    assert(len(q2) == 4)
    assert(t >= 0 and t <= 1)
    if np.all(np.isclose(q1,q2)): return q1
    q1_inv = quaternion_inverse(q1)
    return quaternion_product( q1, quaternion_exp( quaternion_product( q1_inv, q2 ), t ) )

def average_quaternions(quats):
    qs = list(quats)
    if len(qs) == 1: return qs[0]
    while len(qs) > 1:
        if len(qs) % 2 == 1:
            qs[-2] = quaternion_slerp(qs[-2],qs[-1],0.5) # probably better if random
            qs = qs[:-1]
        qs = [quaternion_slerp(q1,q2,0.5) for q1,q2 in zip(qs[::2],qs[1::2])]
    return qs[0]


def rotationAboutAxis(axis,angle, normalizeAxis=True):
    if normalizeAxis: axis = axis / np.linalg.norm(axis)
    angle = angle * 0.5 * np.pi/180
    cos = np.cos( angle )
    sin = np.sin( angle )
    q = [cos] + [sin*x for x in axis]
    return quaternion_to_matrix(q)

# By Chun
def Generate_spanning_vectors(bv1, bv2, bv3, dimensions, buff=5):
    dd = max(dimensions) + 2 * buff

    n1 = round(np.linalg.norm(bv1)/dd) + 1
    n2 = round(np.linalg.norm(bv2)/dd) + 1
    n3 = round(np.linalg.norm(bv3)/dd) + 1

    v1 = np.array(bv1) /n1
    v2 = np.array(bv2) /n2
    v3 = np.array(bv3) /n3

    return v1, v2, v3, n1, n2, n3

def Generate_coordinates(bv1, bv2, bv3, n1, n2, n3, num_copy, origin, replica_index):
    ori_vec = np.array(origin)

    if n1 * n2 * n3 > num_copy:
        check = True
    else:
        check = False

    np.random.seed(42 + replica_index)

    count = 0
    inds = []
    while count < num_copy:
        ind = (np.random.randint(0, n1), np.random.randint(0, n2), np.random.randint(0, n3))
        if not check:
            inds.append(ind)
            count += 1
        elif check:
            if not ind in inds:
                inds.append(ind)
                count += 1
            else:
                while ind in inds:
                    ind = (np.random.randint(0, n1), np.random.randint(0, n2), np.random.randint(0, n3))
                inds.append(ind)
                count += 1

    coors = []
    for ind in inds:
        r1 = (ind[0] - (n1 - 1) * 0.5) * bv1
        r2 = (ind[1] - (n2 - 1) * 0.5) * bv2
        r3 = (ind[2] - (n3 - 1) * 0.5) * bv3
        coors.append((r1 + r2 + r3 + ori_vec).tolist())

    return coors
#Code to here from simplearbd

def readArbdCoords(fname):
    logger.warning('readArbdCoords is deprecated. Please update your code to use read_arbd_coordinates')
    return read_arbd_coordinates(fname)

def read_arbd_coordinates(fname):
    coords = []
    with open(fname) as fh:
        for line in fh:
            coords.append([float(x) for x in line.split()[1:]])
    return np.array(coords)

def readAvgArbdCoords(*args,**kwargs):
    logger.warning('readAvgArbdCoords is deprecated. Please update your code to use read_average_arbd_coordinates')
    return read_average_arbd_coordinates(*args,**kwargs)
    
def read_average_arbd_coordinates(psf,pdb,dcd,rmsd_threshold=3.5, first_frame=0, last_frame=-1, stride=1):
    import MDAnalysis as mda

    usel = mda.Universe(psf, dcd)
    sel = usel.select_atoms("name D*")

    # r0 = ref.xyz[0,ids,:]
    ts = usel.trajectory[last_frame]
    r0 = sel.positions
    pos = []
    for t in range(ts.frame-1,first_frame-1,-stride):
        usel.trajectory[t]
        R,comA,comB = minimizeRmsd(sel.positions,r0)
        r = np.array( [(r-comA).dot(R)+comB for r in sel.positions] )
        rmsd = np.mean( (r-r0)**2 )
        r = np.array( [(r-comA).dot(R)+comB for r in usel.atoms.positions] )
        pos.append( r )
        if rmsd > rmsd_threshold**2:
            break
    t0=t+1
    logger.info(f"Averaging coordinates in {dcd} after frame {t0}")
    
    pos = np.mean(pos, axis=0)
    return pos


def calculate_dimensions_from_cell_vectors(cell_vectors, cell_origin=None, buffer_factor=1.2):
    """
    Calculate simulation box dimensions from cell basis vectors with buffer.
    
    Args:
        cell_vectors: List of 3 cell basis vectors [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]]
        cell_origin: Cell origin coordinates [x,y,z], defaults to [0,0,0]
        buffer_factor: Factor to scale the dimensions for additional buffer space
        
    Returns:
        dimensions: 3D box dimensions as a tuple (x_dim, y_dim, z_dim)
    """
    if cell_vectors is None:
        return None
        
    # Make sure we have proper cell vectors
    if len(cell_vectors) != 3:
        raise ValueError("Expected 3 cell basis vectors, got {}".format(len(cell_vectors)))
    
    # Convert to numpy arrays for easier manipulation
    vectors = [np.array(v) for v in cell_vectors]

    # Check that vectors are orthogonal and aliged with x,y,z axis
    if not all( v[i] == np.linalg.norm(v) for i,v in vectors ):
        raise NotImplementedError('Cell vectors not aligned along x, y and z axes are not yet supported')
    
    # Calculate maximum extent along each dimension
    dimensions = [0, 0, 0]
    for i in range(3):
        for v in vectors:
            dimensions[i] += abs(v[i])
    
    # Apply buffer factor
    dimensions = [dim * buffer_factor for dim in dimensions]
    
    return tuple(dimensions)

def unit_quat_conversions():
    for axis in [[0,0,1],[1,1,1],[1,0,0],[-1,-2,0]]:
        for angle in np.linspace(-180,180,10):
            R = rotationAboutAxis(axis, angle)
            R2 = quaternion_to_matrix( quaternion_from_matrix( R ) )
            if not np.all( np.abs(R-R2) < 0.01 ):
                import pdb
                pdb.set_trace()
                quaternion_to_matrix( quaternion_from_matrix( R ) )


if __name__ == "__main__":
    unit_quat_conversions()
            
