from scipy.spatial import KDTree
import numpy as np


## I'm not convinced this is that great
def find_shape_based_sites(fine_positions, N_cg,
                           num_steps = None, learning_schedule=None,
                           weights = None,
                           seed = 1234):

    rng = np.random.default_rng(seed=seed)
    
    n_fg = len(fine_positions)

    if weights is None:
        weights = np.ones(n_fg)
    weights = np.array(weights)
    weights = weights / np.sum(weights)    

    if num_steps is None:
        num_steps = 200*N_cg

    # p.collapsedPosition()
    # r_fg = [p.position for p in fine]
    r_fg = fine_positions
    
    ## Initialize CG sites
    r_cg = r_fg[rng.choice( n_fg, size=N_cg, replace=False, p=weights )]
    if not np.all(np.isfinite(r_cg)):
        raise Exception

    if learning_schedule is None:
        e0,e1 = 0.3,0.05
        l0,l1 = 0.2*N_cg,0.01
        
        epsilon = lambda s: e0*(e1/e0)**(s/num_steps)
        lambda_  = lambda s: l0*(l1/l0)**(s/num_steps)

    ## choose random fine particle for each step
    rs = r_fg[rng.choice( n_fg, size=num_steps, p=weights )]
    
    for step,r in enumerate(rs,1):
        dr0  = (r[None,:]-r_cg)
        dr0_sq = (dr0**2).sum(axis=-1)

        ## number of CG sites closer to FG site
        k = np.array([(dr0_sq < x).sum() for x in dr0_sq])
        if not np.all(np.isfinite(k)):
            raise Exception
        
        learn_rate = epsilon(step) * np.exp(-k/lambda_(step))

        if not np.all(np.isfinite(learn_rate)):
            raise Exception

        r_cg = r_cg + learn_rate[:,None] * dr0
                
    return r_cg


def get_particle_assignments( fine_sites, coarse_sites, max_distance=20 ):

    t_fg = KDTree(  fine_sites  )
    t_cg = KDTree( coarse_sites )

    coo = t_fg.sparse_distance_matrix( t_cg, output_type='coo_matrix', max_distance=max_distance )

    csr = coo.tocsr()

    def _nonzero_row_argmin( csr ):
        result = []
        for i in range(csr.shape[0]):
            sl = slice(csr.indptr[i], csr.indptr[i+1])
            if len(csr.data[sl]) == 0:
                raise Exception("Some fine particles too far from coarse sites to assign")
            else:
                idx = np.argmin(csr.data[sl])
                j = csr.indices[sl][idx] # column index
                result.append(j)
        return np.array(result, dtype=int)


    assignments = _nonzero_row_argmin(csr)
    assert(len(assignments) == len(fine_sites))
    return assignments
