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
    assert(len(q) == 4)

    ## It looks like the wikipedia article I used employed a less common convention for q (see below
    ## http://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rotation_matrix_.E2.86.94_quaternion
    # q1,q2,q3,q4 = q
    # R = [[1-2*(q2*q2 + q3*q3),    2*(q1*q2 - q3*q4),    2*(q1*q3 + q2*q4)],
    #      [  2*(q1*q2 + q3*q4),  1-2*(q1*q1 + q3*q3),    2*(q2*q3 - q1*q4)],
    #      [  2*(q1*q3 - q2*q4),    2*(q1*q4 + q2*q3),  1-2*(q2*q2 + q1*q1)]]

    q = q / np.linalg.norm(q)
    q0,q1,q2,q3 = q
    R = [[1-2*(q2*q2 + q3*q3),    2*(q1*q2 - q3*q0),    2*(q1*q3 + q2*q0)],
         [  2*(q1*q2 + q3*q0),  1-2*(q1*q1 + q3*q3),    2*(q2*q3 - q1*q0)],
         [  2*(q1*q3 - q2*q0),    2*(q1*q0 + q2*q3),  1-2*(q2*q2 + q1*q1)]]

    return np.array(R)

def quaternion_from_matrix( R ):
    R=R.T
    q = np.empty(4)
    if R[2,2] < 0:
        if R[0,0] > R[1,1]:
            trace = 1.0 + R[0,0] - R[1,1] - R[2,2]
            s = 2.0 * np.sqrt(trace)
            if R[1,2] < R[2,1]: s = -s
            q[0] = (R[1,2] - R[2,1]) / s
            q[1] = 0.25 * s
            q[2] = (R[0,1] + R[1,0]) / s
            q[3] = (R[2,0] + R[0,2]) / s
            if np.isclose(trace,1) and np.all(np.isclose([x for i,x in enumerate(q) if i != 1],0)):
                q[1] = 1
        else:
            trace = 1.0 - R[0,0] + R[1,1] - R[2,2]
            s = 2.0 * np.sqrt(trace)
            if R[2,0] < R[0,2]: s = -s
            q[0] = (R[2,0] - R[0,2]) / s
            q[1] = (R[0,1] + R[1,0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1,2] + R[2,1]) / s
            if np.isclose(trace,1) and np.all(np.isclose([x for i,x in enumerate(q) if i != 2],0)):
                q[2] = 1
    else:
        if R[0,0] < -R[1,1]:
            trace = 1.0 - R[0,0] - R[1,1] + R[2,2]
            s = 2.0 * np.sqrt(trace)
            if R[0,1] < R[1,0]: s = -s
            q[0] = (R[0,1] - R[1,0]) / s
            q[1] = (R[2,0] + R[0,2]) / s
            q[2] = (R[1,2] + R[2,1]) / s
            q[3] = 0.25 * s
            if np.isclose(trace,1) and np.all(np.isclose([x for i,x in enumerate(q) if i != 3],0)):
                q[3] = 1
        else:
            trace = 1.0 + R[0,0] + R[1,1] + R[2,2]
            s = 2.0 * np.sqrt(trace)
            q[0] = 0.25 * s
            q[1] = (R[1,2] - R[2,1]) / s
            q[2] = (R[2,0] - R[0,2]) / s
            q[3] = (R[0,1] - R[1,0]) / s
            if np.isclose(trace,1) and np.all(np.isclose([x for i,x in enumerate(q) if i != 0],0)):
                q[0] = 1
    assert( q[0] >= 0 )
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
    q1_inv = quaternion_inverse(q1)
    return quaternion_product( q1, quaternion_exp( quaternion_product( q1_inv, q2 ), t ) )

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
    logger.info("Averaging coordinates in %s after frame %d" % (dcd, t0) )
    
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
            
