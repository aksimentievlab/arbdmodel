from pathlib import Path
from chrispy.grids import writeDx
import numpy as np
from scipy import signal

""" Script for writing dx file with half-harmonic confinement potential """


""" Tune these parameters to set the boundary conditions """
def write_confine( radius=100 ):

    outfile=f'confine-{radius}.dx'
    if Path(outfile).exists(): return
    
    k = 1                           # Spring constant [kcal/mol/AA^2]
    # radius=800

    x0 = y0 = -radius - 50
    x1 = y1 = -x0
    z0,z1 = x0,x1

    dx = dy = dz = 2

    assert( x1 > x0 )
    assert( y1 > y0 )
    assert( z1 > z0 )

    """ Create grid axes """
    x,y,z = [np.arange( a-res/2, b+res/2, res )
             for a,b,res in zip((x0,y0,z0),(x1,y1,z1),(dx,dy,dz))]
    # x = np.arange( -100, 100, dx ) # alternatively, be explicit

    # assert( x[0] == -x[-1] )

    X,Y,Z = np.meshgrid(x,y,z,indexing='ij')      # create meshgrid for making potential
    R = np.sqrt(X**2 + Y**2 + Z**2) 


    """ Create the potential, adding 0.5 k deltaX**2 for each half plane """
    pot = np.zeros( X.shape )


    ids = R > radius
    pot[ids] = 0.5*k*(R[ids]-radius)**2

    ids = R > radius + 25
    pot[ids] = 0.5*k*25**2 + 0.5*k*25**2*(R[ids]-radius-25) # switch to linear potential

    """ Write the dx file """
    writeDx(outfile, pot,
             delta=(dx,dy,dz),
             origin=(x[0],y[0],z[0]))


for r in np.arange(200,210,25):
    print(r)
    write_confine(int(r))
