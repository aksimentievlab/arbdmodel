# SimpleARBD/Accessory_routines_for_ARBD.py, using grid.py
from __future__ import absolute_import, print_function
import numpy as np 
from scipy import signal
import math
import os,sys
from .grid import writeDx, gaussian_kernel, loadGrid

def info(*obj):
    print('INFO: ',obj , file=sys.stderr)

def Get_damping_coefficients(hydroproFile, massFile, inertiaFile, outFile):
    """Calculate damping coefficients"""
    lineNum = 1
    with open(hydroproFile, 'r') as fin:
        ## skip 49 lines
        while lineNum <= 48:
            fin.readline()
            lineNum += 1

        ## read 3 lines
        Dx = float(fin.readline().strip().split()[0])
        Dy = float(fin.readline().strip().split()[1])
        Dz = float(fin.readline().strip().split()[2])

        ## skip 2 lines
        fin.readline()
        fin.readline()

        ## read 3 lines
        Rx = float(fin.readline().strip().split()[3])
        Ry = float(fin.readline().strip().split()[4])
        Rz = float(fin.readline().strip().split()[5])

    with open(massFile ,'r') as fin:
        mass = float(fin.readline().strip())
    with open(inertiaFile, 'r') as fin:
        inertia = [float(x) for x in fin.readline().strip().split()]

    ## convert
    # units "(295 k K) / (( cm^2/s) *  amu)" "1/ns"
    Dx, Dy, Dz = [24.527692/(x*mass) for x in [Dx,Dy,Dz]]

    # units "(295 k K) / ((1 /s) *  amu AA^2)" "1/ns"
    Rx, Ry, Rz = [2.4527692e+17 / (x*mass) for x, mass in zip([Rx,Ry,Rz],inertia)]

    with open(outFile, 'w') as fout:
        fout.write(' '.join([str(Dx), str(Dy), str(Dz)]) + '\n')
        fout.write(' '.join([str(Rx), str(Ry), str(Rz)]))

def Fix_charge(inFile, outFile, netChargeFile):
    """Fix charge distribution"""
    # First fix scientific notation in file
    cmd_in = "sed -r 's/^([0-9]+)e/\1.0e/g; s/ ([0-9]+)e/ \1.0e/' " + inFile + " > fix_charge_temp0.dx"
    os.system(cmd_in)
    cmd_in = "sed -r 's/^(-[0-9]+)e/\1.0e/g; s/ (-[0-9]+)e/ \1.0e/' fix_charge_temp0.dx > fix_charge_temp1.dx"
    os.system(cmd_in)

    resolution = 2

    with open(netChargeFile, 'r') as fout:
        netCharge = float(fout.readline().strip())

    # Load data
    grid, origin, delta = loadGrid('fix_charge_temp1.dx')
    grid = grid * resolution**3

    # Apply upper and lower bounds
    ids = np.where(np.abs(grid[:]) > 0.01)

    numPoints = np.size(ids)
    info(np.sum(grid), numPoints, np.sum(grid)/numPoints)

    # Remove excess charge (in loop due to machine error)
    while np.abs(np.sum(grid) - netCharge) > 0.0001:
        grid[ids] = grid[ids] + (netCharge-np.sum(grid))/numPoints
        #info(np.sum(grid), numPoints, np.sum(grid)/numPoints)

    info("Final charge", np.sum(grid))

    # Write output using writeDx
    writeDx(outFile, grid, origin, [delta, delta, delta])

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

def Find_segments_num(dimensions, threshold=300):
    """Find number of segments needed"""
    in_xyz = [float(elm) for elm in dimensions]
    segments = [math.ceil(elm / threshold) for elm in in_xyz]
    return segments[0], segments[1], segments[2]

def Find_boundary_resolution(cellBasisVector1=[10,0,0],
                           cellBasisVector2=[0,10,0],
                           cellBasisVector3=[0,0,10],
                           resolution=2):
    """Find appropriate boundary resolution"""
    min_PBC_length = min([max(cellBasisVector1), max(cellBasisVector2), max(cellBasisVector3)])
    if min_PBC_length < resolution:
        dx = round(min_PBC_length / 2)
    else:
        dx = resolution

    n1 = round(np.linalg.norm(cellBasisVector1)/dx)
    n2 = round(np.linalg.norm(cellBasisVector2)/dx)
    n3 = round(np.linalg.norm(cellBasisVector3)/dx)

    return dx, n1, n2, n3

# Other functions remaining the same as they don't use GridData...

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