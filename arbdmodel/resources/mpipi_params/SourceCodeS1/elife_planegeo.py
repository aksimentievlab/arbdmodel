#!/usr/bin/python

import string
from sys import argv,stdout
from os import popen,system
from os.path import exists,basename
#from amino_acids import longer_names
from math import log, sqrt, pi
from numpy import cross, array, dot, vdot, arccos
import numpy as np
import copy

def unit_vector(vector):
    #Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    #Returns the angle in radians between vectors 'v1' and 'v2'::
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return angle

def read_pdb( file ): # Returns dictionary D{chain+"."res number} = [ aa, {atoms} = [x,y,z] ]
    pdb = open( file ).readlines()
    pdbdict = {}
    for i in pdb:
        line = i.split()

        if len(line) > 5:
            if i[:4] == "ATOM":
                atom = i[13:16].split()[0]
                res = i[17:20]
                chain = i[21:22]
                x = float(i[30:38])
                y = float(i[38:46])
                z = float(i[46:54])
                resN = int(i[22:26])

                atype = i[77:78]

                key = chain+"."+str(resN)

                if rgroups.has_key(res) and atype != "H":
                    #if atom.find('H') == -1 or atom == "NH1" or atom == "NH2":

                        if pdbdict.has_key(key) == False:
                            pdbdict[key] = [res, {}]

                        if pdbdict[key][1].has_key(atom) == False:
                            pdbdict[key][1][atom] = array([x,y,z])
    return pdbdict
  
def read_sym( file ): # Returns dictionary D{chain+"."res number} = [ aa, {atoms} = [x,y,z] ]
    pdb = open( file ).readlines()
    pdbdict = {}
    for i in pdb:
        line = i.split()

        onchain = 0

        if len(line) > 5:
            if i[:4] == "ATOM":
                atom = i[13:16].split()[0]
                res = i[17:20]
                chain = i[21:22]+str(onchain)
                x = float(i[30:38])
                y = float(i[38:46])
                z = float(i[46:54])
                resN = int(i[22:26])

                key = chain+"."+str(resN)

                if rgroups.has_key(res):
                    
                    if pdbdict.has_key(key):
                        if pdbdict[key][1].has_key(atom):
                            pa = pdbdict[key][1][atom]
                            distance = sqrt( (pa[0] - x)**2 + (pa[1] - y)**2 + (pa[2] - z)**2 )
                            if distance > 20.0:
                                onchain += 1
                                chain = i[21:22]+str(onchain)
                                key = chain+"."+str(resN)

                    if pdbdict.has_key(key) == False:
                        pdbdict[key] = [res, {}]

                    if pdbdict[key][1].has_key(atom) == False:
                        pdbdict[key][1][atom] = array([x,y,z])
    return pdbdict

 
def is_complete( pdbdict ):#Annotates dictionary for removing incomplete residues (ie: unmodeled sidechains due to missing density)
    for key in pdbdict.keys():
        complete = True
        aa = pdbdict[key][0]
        for a in rgroups[aa]:
            if pdbdict[key][1].has_key(a) == False:
                complete = False

        if aa != "A" and aa != "G" and aa != "C" and aa != "U":
            for a in rgroups["BB"]:
                if pdbdict[key][1].has_key(a) == False:
                    complete = False
            if pdbdict[key][1].has_key('N') == False:
                complete = False
        
        pdbdict[key].append(complete)


def peptide_details(res1,res2):#N, CA, C, O, Np, CAp, Cp):
    
    N = res1["N"]
    CA = res1["CA"]
    C = res1["C"]
    O = res1["O"]
    Np = res2["N"]
    CAp = res2["CA"]
    Cp = res2["C"]

    psi = torsion_angle(N,CA,C,Np)
    #while psi < 0:
    #    psi += pi

    phi_p1 = torsion_angle(C,Np,CAp,Cp)
    #while phi_p1 < 0:
    #    phi_p1 += pi

    omega = torsion_angle(CA,C,Np,CAp)
    #while omega < 0:
    #    omega += pi

    vecs = make_axis_vectors(O, C, CA)

    normvec = vecs[1]

    center = (CA+C+O+Np+CAp)/5.0

    alist = [C,O,Np]

    return [np.degrees(psi), np.degrees(phi_p1), np.degrees(omega), normvec, center, alist]

def compare_res_to_BB( bb, res ):

    rg = rgroups[res[0]]#rgroups[res2[0]]
    ag = agroups[res[0]]

    A = res[1][rg[0]]
    B = res[1][rg[1]]
    C = res[1][rg[2]]

    rvecs = make_axis_vectors(A, B, C) 

    #dot_toBB = vdot(r1vecs[0], r2vecs[0]) # A-B bond vectors
    dot_orthog = (vdot(rvecs[1], bb[3])) # vectors perpendicular to plane
    #NOTE: Absolute value because side chain plane vector is based on degenerate atoms

    #dot_90deg = vdot(r1vecs[2], r2vecs[2]) # vectors 90 degrees to A-B bond in plane

    center = array([0.0,0.0,0.0])
    for xi in rgroups[res[0]]:
        center += res[1][xi]
    center /= float(len(rgroups[res[0]]))

    vd = center - bb[4]
    vdlen = sqrt(vd[0]**2 + vd[1]**2 + vd[2]**2)
    vd[0] = vd[0] / vdlen
    vd[1] = vd[1] / vdlen
    vd[2] = vd[2] / vdlen

    dot_rcomp1 = vdot(vd, bb[3])# # res-bb vector orientation to bb_orthog
    #dot_rcomp2 = vdot(vd, r2vecs[1]) # res1B-res2B vector orientation to res2_orthog 

    min_distance = -1.0
    for xi in rg:
        #for yi in rg2:#res2[1].keys():
        dp = res[1][xi] - bb[4]#res2[1][yi]
        distance = sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)

        if distance < min_distance or min_distance == -1.0:
            min_distance = distance

    abmin_pi_distance = -1.0
    pairmatch = 0#[0,0]
    rnvec = rvecs[1]
    bnvec = bb[3]

    udlist1 = []
    udlist2 = []
    plen = 1.7
    for y in range(len(bb[5])):#yi in bb[5]:
        yi = bb[5][y]
        bb_up = yi + (bnvec * plen)
        bb_down = yi - (bnvec * plen)
        udlist1.append([bb_up,bb_down])

    for x in range(len(ag)):#xi in rg:
        xi = ag[x]
        res_up = res[1][xi] + (rnvec * plen)
        res_down = res[1][xi] - (rnvec * plen)
        udlist2.append([res_up,res_down])

    pairmatch = calc_match(udlist1,udlist2)


    return [dot_orthog, dot_rcomp1, min_distance, abmin_pi_distance, pairmatch[0], pairmatch[1]]

def compare_BB_to_BB( bb1, bb2 ):

    bnvec1 = bb1[3]
    bnvec2 = bb2[3]

    dot_orthog = (vdot(bnvec1, bnvec2)) # vectors perpendicular to plane

    udlist1 = []
    udlist2 = []
    plen = 1.7
    for y in range(len(bb1[5])):#yi in bb[5]:
        yi = bb1[5][y]
        bb1_up = yi + (bnvec1 * plen)
        bb1_down = yi - (bnvec1 * plen)
        udlist1.append([bb1_up,bb1_down])

    for x in range(len(bb2[5])):#xi in rg:
        xi = bb2[5][x]
        bb2_up = xi + (bnvec2 * plen)
        bb2_down = xi - (bnvec2 * plen)
        udlist2.append([bb2_up,bb2_down])

    pairmatch = calc_match(udlist1,udlist2)


    return [dot_orthog, pairmatch[0], pairmatch[1]]


def torsion_angle(A, B, C, D):
        
    b1 = C - D
    b2 = B - C
    b3 = A - B

    if b2[0] != 0 and b2[1] != 0 and b2[2] != 0:
    #if abs(b2[0]) > 0 or abs(b2[1]) > 0 or abs(b2[2]) > 0:
        left = dot(cross( cross(b1,b2), cross(b2,b3) ), b2/abs(b2))
        right = dot(cross(b1,b2),cross(b2,b3))
        torangle = np.arctan2(left,right) 
        return torangle
    else:
        return 0.0

def make_axis_vectors(A, B, C): # For np.array([x,y,z]) of connected atoms A->B->C
    
    Avec = A-B # vector along A->B bond
    
    r1cen = A - B#pdbdict[p1][2][rg1[0][0]] - pdbdict[p1][2][rg1[0][1]] 
    r2cen = C - B#pdbdict[p1][2][rg1[1][1]] - pdbdict[p1][2][rg1[0][1]]

    Nvec = cross(r1cen, r2cen) # vector parralel to plane

    r1cen = A - B
    r2cen = Nvec

    Cvec = cross(r1cen, r2cen) # 90 degree angle to Avec, in plane with A,B,C

    Alen = sqrt(Avec[0]**2+Avec[1]**2+Avec[2]**2)
    Avec /= Alen

    Nlen = sqrt(Nvec[0]**2+Nvec[1]**2+Nvec[2]**2)
    Nvec /= Nlen

    Clen = sqrt(Cvec[0]**2+Cvec[1]**2+Cvec[2]**2)
    Cvec /= Clen

    return [Avec, Nvec, Cvec]
 
#   The distance between the second atoms in the rgroups lists
#   which for this function we can just assume is a random atom
#   specific to the residue type in question
def rough_distance( res1, res2 ):
    dp = res1[1][rgroups[res1[0]][1]] - res2[1][rgroups[res2[0]][1]]
    distance = sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)

    return distance

#   The distance between calpha atoms
def CA_distance( res1, res2 ):
    dp = res1[1]["CA"] - res2[1]["CA"]
    distance = sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)

    return distance

#   for both of two residues, gets the a list of points representing the 1.7 angstrom above the plane coordinates
#   closest to the center of mass for the other sidechain
def get_closepoints( udlist1, udlist2, center1, center2):

    l1 = []
    for i in udlist1:
        dp = center2 - i[0]
        distanceU = sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)
        dp = center2 - i[1]
        distanceD = sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)

        if distanceU < distanceD:
            l1.append(i[0])
        else:
            l1.append(i[1])

    l2 = []
    for i in udlist2:
        dp = center1 - i[0]
        distanceU = sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)
        dp = center1 - i[1]
        distanceD = sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)

        if distanceU < distanceD:
            l2.append(i[0])
        else:
            l2.append(i[1])

    return [l1,l2]

#   Finds the smallest number of atoms in two sidechains that satisfy the match criteria
#   where a match is when points 1.7 angstroms above the plane for each atom (ie: the carbon VDW surface) 
#   are within 1.5 angstroms of each other
def calc_match( udlist1, udlist2 ):
    
    return_points = False

    short = udlist1
    long = udlist2

    if len(udlist1) > len(udlist2):
        short = udlist2
        long = udlist1

    pointlist = []

    ntotal = 0.0
    nmatch = 0.0
    allmatch = []
    for n1 in range(len(long)):#ud1 in long:
        ud1 = long[n1]
        ntotal += 1.0

        vec1 = unit_vector(ud1[0] - ud1[1])

        c1 = (ud1[0]+ud1[1])
        c1[0] /= 2.0
        c1[1] /= 2.0
        c1[2] /= 2.0


        for n2 in range(len(short)):#short:
            min_dis = -1
            ud2 = short[n2]

            vec2 = unit_vector(ud2[0] - ud2[1])

            c2 = (ud2[0]+ud2[1])
            c2[0] /= 2.0
            c2[1] /= 2.0
            c2[2] /= 2.0

            #v1to2 = unit_vector(c2 - c1)
            #v2to1 = unit_vector(c1 - c2)

            #d1to2 = abs(vdot(v1to2, vec1))
            #d2to1 = abs(vdot(v2to1, vec2))

            #if d1to2 >= 0.5 and d2to1 >= 0.5:
            dp = ud1[0] - ud2[0]
            distance = sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)
            if distance < min_dis or min_dis == -1.0:
                min_dis = distance
                store_ud = [ud1[0],ud2[0]]

            dp = ud1[0] - ud2[1]
            distance = sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)
            if distance < min_dis:
                min_dis = distance
                store_ud = [ud1[0],ud2[1]]


            dp = ud1[1] - ud2[0]
            distance = sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)
            if distance < min_dis:
                min_dis = distance
                store_ud = [ud1[1],ud2[0]]


            dp = ud1[1] - ud2[1]
            distance = sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)
            if distance < min_dis:
                min_dis = distance
                store_ud = [ud1[1],ud2[1]]

            if min_dis < 1.5:# and d1to2 >= 0.5 and d2to1 >= 0.5:
                allmatch.append([min_dis, n1, n2, store_ud])
 
    allmatch.sort()
    allmatch.reverse()

    taken1 = {}
    taken2 = {}
    for a in allmatch:
        if taken1.has_key(a[1]) == False and taken2.has_key(a[2]) == False:
            pointlist.append(a[3])
        taken1[a[1]] = True
        taken2[a[2]] = True
            
    if len(taken1.keys()) < len(taken2.keys()):
        nmatch = len(taken1.keys())
    else:
        nmatch = len(taken2.keys())

    center = array([ 0.0,  0.0,  0.0])
    if len(allmatch) > 0:

        for a in allmatch:
            center += a[3][0]
            center += a[3][1]
        center /= float(len(allmatch))*2.0

        if abs(center[0]) < 0.1 and allmatch[0][3][0][0] > 5.0:
            print 1/0

    if return_points == False:
        return [nmatch, center]
    else:
        return pointlist#ha ha it's pointlist

def compare_residues( res1, res2 ):
    return calc_residue_match(res1,res2,"")


#   Finds the smallest number of atoms in two sidechains that satisfy the match criteria
#   where a match is when points 1.7 angstroms above the plane for each atom (ie: the carbon VDW surface) 
#   are within 1.5 angstroms of each other
#
#   Default return style puts out a list of other useful variables
#   [dot_orthog, dot_rcomp1, dot_rcomp2, abmin_distance, num_match]
#
#       dot_orthog
#           dot product orientation between the two planes (1.0 = identical, 0.0 = 90 degree angle) 
#           (you can get the angle between the planes by putting this into acos(x))
#
#       dot_rcomp1
#           dot product between residue 1's above-the-plane unit vector and the unit vector between the 
#           two sidechain centers of mass. So this describes residue2's position relative to residue1's plane
#           with 1.0 = directly above the plane and 0.0 = alongside the plane
#
#       dot_rcomp2
#           same as above, but using residue 2's above-the-plane unit vector, so it describes residue1's position
#           relative to residue2 this time.
#
#       abmin_distance
#           smallest distance between two atoms
# 
#       num_match
#           the number of atoms satisfying the match criteria
#
#   the "style" option allows you to switch behavior.
#
#   match = just returns num_match
#
#   points = just spitting out a list of points representing the 1.7 angstrom above the plane coordinates closest 
#   to the other system for both sidechains... (no distance cutoff)
def calc_residue_match( res1, res2, style ):
    rg1 = rgroups[res1[0]]
    rg2 = rgroups[res2[0]]

    A1 = res1[1][rg1[0]]
    B1 = res1[1][rg1[1]]
    C1 = res1[1][rg1[2]]

    r1vecs = make_axis_vectors(A1, B1, C1) 

    A2 = res2[1][rg2[0]]
    B2 = res2[1][rg2[1]]
    C2 = res2[1][rg2[2]]

    r2vecs = make_axis_vectors(A2, B2, C2) 

    #dot_toBB = vdot(r1vecs[0], r2vecs[0]) # A-B bond vectors
    dot_orthog = (vdot(r1vecs[1], r2vecs[1])) # vectors perpendicular to plane
    #dot_90deg = vdot(r1vecs[2], r2vecs[2]) # vectors 90 degrees to A-B bond in plane

    center1 = array([0.0,0.0,0.0])
    for xi in rgroups[res1[0]]:
        center1 += res1[1][xi]
    center1 /= float(len(rgroups[res1[0]]))
        
    center2 = array([0.0,0.0,0.0])
    for xi in rgroups[res2[0]]:
        center2 += res2[1][xi]
    center2 /= float(len(rgroups[res2[0]]))

    #vd = res1[1][rg1[1]] - res2[1][rg2[1]]
    vd = center1 - center2

    vdlen = sqrt(vd[0]**2 + vd[1]**2 + vd[2]**2)
    vd[0] = vd[0] / vdlen
    vd[1] = vd[1] / vdlen
    vd[2] = vd[2] / vdlen

    dot_rcomp1 = vdot(vd, r1vecs[1]) # res1B-res2B vector orientation to res1_orthog 
    dot_rcomp2 = vdot(vd, r2vecs[1]) # res1B-res2B vector orientation to res2_orthog 

    abmin_distance = -1.0
    for xi in rg1:#res1[1].keys():
        for yi in rg2:#res2[1].keys():
            dp = res1[1][xi] - res2[1][yi]
            distance = sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)

            if distance < abmin_distance or abmin_distance == -1.0:
                abmin_distance = distance

    r1nvec = r1vecs[1]
    r2nvec = r2vecs[1]

    if agroups.has_key(res1[0]) and agroups.has_key(res2[0]):
        ag1 = agroups[res1[0]]
        ag2 = agroups[res2[0]]

        rg1_nlist = []
        for xi in ag1:
            x = res1[1][xi]
            plen = 1.7
            u = x + (r1nvec*plen)
            d = x - (r1nvec*plen)

            rg1_nlist.append([u,d])

        rg2_nlist = []
        for xi in ag2:
            x = res2[1][xi]
            plen = 1.7
            u = x + (r2nvec*plen)
            d = x - (r2nvec*plen)

            rg2_nlist.append([u,d])

    

    if style == "points":
        closepoints = get_closepoints(rg1_nlist, rg2_nlist, center1, center2)
        return closepoints
        
    num_match = calc_match( rg1_nlist, rg2_nlist )
    if style == "match":
        return num_match

    return [dot_orthog, dot_rcomp1, dot_rcomp2, abmin_distance, num_match[0], num_match[1]]



rgroups = {}

#No SP2s, these atom sets are random planes I was using as a control in the hoary days of olde
#but I ended up using the keys in this dictionary as my standin list of amino acids so here they are
rgroups["ALA"] = ["C", "CB", "CA"]
rgroups["LEU"] = ["CD1", "CG", "CD2", "CB"]
rgroups["VAL"] = ["CG1", "CB", "CG2"]
rgroups["ILE"] = ["CD1", "CG1", "CB", "CG2"]
rgroups["MET"] = ["CE", "SD", "CG", "CB"]
rgroups["CYS"] = ["SG", "CB", "CA"]
rgroups["SER"] = ["OG", "CB", "CA"]
rgroups["THR"] = ["OG1", "CB", "CG2"]
rgroups["LYS"] = ["NZ", "CE", "CD", "CG", "CB"]

#This one exists to prevent index errors when testing against the GLY sidechain, which... uh, I'm not sure how that came up...
rgroups["GLY"] = ["O", "C", "CA"]

#Sidechain pi-system planes are computed based on these atom lists, where the plane is defined based on the first three atoms
rgroups["ASN"] = ["CB", "CG", "OD1", "ND2"]
rgroups["GLN"] = ["CG", "CD", "OE1", "NE2"]
rgroups["ASP"] = ["CB", "CG", "OD1", "OD2"]
rgroups["GLU"] = ["CG", "CD", "OE1", "OE2"]
rgroups["ARG"] = ["NE", "CZ", "NH1", "NH2"]
rgroups["PRO"] = ["CA", "N", "CD", "CG", "CB"]
rgroups["HIS"] = ["CB", "CG", "ND1", "CD2", "CE1", "NE2"]
rgroups["PHE"] = ["CB","CG","CD1","CD2","CE1","CE2","CZ"]
rgroups["TYR"] = ["CB","CG","CD1","CD2","CE1","CE2","CZ"]
rgroups["TRP"] = ["CB", "CG", "CD2", "CD1", "NE1", "CE2", "CZ2", "CH2", "CZ3", "CE3"]

#(one of the functions checks if the residue in question has all the atoms it needs before trying to compute
# pi-system planes and stuff, and when that function looks at backbones it uses this for the first residue and
# then a special N check against the second residue... I don't know man, this system kind of grew organically here...)
rgroups["BB"] = ["O", "C", "CA"]

agroups = {}

#SP2 Sidechain Atoms (basically rgroups minus the atom-closest-to-backbone, which in rgroups is used for orienting the planes)
agroups["ASN"] = ["CG", "OD1", "ND2"]
agroups["GLN"] = ["CD", "OE1", "NE2"]
agroups["ASP"] = ["CG", "OD1", "OD2"]
agroups["GLU"] = ["CD", "OE1", "OE2"]
agroups["ARG"] = ["NE", "CZ", "NH1", "NH2"]
agroups["HIS"] = ["CG", "ND1", "CD2", "CE1", "NE2"]
agroups["PHE"] = ["CG","CD1","CD2","CE1","CE2","CZ"]
agroups["TYR"] = ["CG","CD1","CD2","CE1","CE2","CZ"]
agroups["TRP"] = ["CG", "CD2", "CD1", "NE1", "CE2", "CZ2", "CH2", "CZ3", "CE3"]


#VDW radius cutoff
diCUT = float(4.9)
dCUT = "4.9"

#Minimum number of atom-pairs required to define a contact
CUT = "2"
CUTi = 2
CUTf = 2.0

#Counts the minimum number of unique atom-pairs under the VDW radius, for coordinate lists l1 and l2
def mindist_count( l1, l2):
    
    xdone = {}
    ydone = {}
    for x in l1:

        distance = 0.0
        for y in l2:
            dp = x - y
            distance = sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)
            
            if distance <= dCUT:
                xdone[str(x)] = 0
                ydone[str(y)] = 0

            #Stop counting if a distance measurement makes a contact impossible
            if distance >= 28.0:
                break
        if distance >= 28.0:
            break

    mincount = len(xdone.keys())
    if len(ydone.keys()) < mincount:
        mincount = len(ydone.keys())

    return mincount

#Sets up sp2 atom coordinate lists for backbone to backbone tests
def bbbb_distance( bb1, bbn1, bb2, bbn2 ):
    bb1list = []
    bb2list = []

    bb1list.append(bb1[1]["C"])
    bb1list.append(bb1[1]["O"])
    bb1list.append(bbn1[1]["N"])

    bb2list.append(bb2[1]["C"])
    bb2list.append(bb2[1]["O"])
    bb2list.append(bbn2[1]["N"])

    return mindist_count( bb1list, bb2list )

#Sets up sp2 atom coordinate lists for sidechain to backbone tests
def bbsc_distance( bb, bbn, sc ):
    bblist = []
    sclist = []

    bblist.append(bb[1]["C"])
    bblist.append(bb[1]["O"])
    bblist.append(bbn[1]["N"])

    ag = agroups[sc[0]]
    
    for a in ag:
        sclist.append(sc[1][a])

    return mindist_count( bblist, sclist )

#Sets up sp2 atom coordinate lists for sidechain to sidechain tests
def scsc_distance( sc1, sc2 ):
    sc1list = []
    sc2list = []

    ag = agroups[sc1[0]]
    for a in ag:
        sc1list.append(sc1[1][a])

    ag = agroups[sc2[0]]
    for a in ag:
        sc2list.append(sc2[1][a])

    return mindist_count( sc1list, sc2list )

#Tests backbone to backbone contact
# returns:
#   hits[0] = VDW contact (0 = False, 1 = True)
#   hits[1] = Surface Contact (0 = False, 1 = True)
#   hits[2] = Planar Contact (0 = False, 1 = True)
def allcompare_bb2bb( bb1, bbn1, bb2, bbn2 ):
    hits = [0,0,0]

    distance = bbbb_distance( bb1, bbn1, bb2, bbn2 )
    if distance >= CUTi:
        hits[0] = 1
    
        peptide1 = peptide_details(bb1[1], bbn1[1])
        peptide2 = peptide_details(bb2[1], bbn2[1])
        plcomp = compare_BB_to_BB( peptide1, peptide2 )

        if int(plcomp[1]) >= CUTi:
            hits[1] = 1

            if abs(plcomp[0]) >= 0.8:#planar dot product between pi systems
                hits[2] = 1
                hits.append(plcomp[2])

    return hits

#Tests sidechain to backbone contact
# returns:
#   hits[0] = VDW contact (0 = False, 1 = True)
#   hits[1] = Surface Contact (0 = False, 1 = True)
#   hits[2] = Planar Contact (0 = False, 1 = True)
def allcompare_bb2sc( bb, bbn, sc ):
    hits = [0,0,0]

    distance = bbsc_distance( bb, bbn, sc )
    if distance >= CUTi:
        hits[0] = 1
    
        peptide = peptide_details(bb[1], bbn[1])
        plcomp = compare_res_to_BB( peptide, sc )

        if int(plcomp[4]) >= CUTi:
            hits[1] = 1

            if abs(plcomp[0]) >= 0.8:#planar dot product between pi systems
                hits[2] = 1
    return hits

#Tests sidechain to sidechain contact
# returns:
#   hits[0] = VDW contact (0 = False, 1 = True)
#   hits[1] = Surface Contact (0 = False, 1 = True)
#   hits[2] = Planar Contact (0 = False, 1 = True)
def allcompare_sc2sc( sc1, sc2 ):
    hits = [0,0,0]

    distance = scsc_distance( sc1, sc2 )
    if distance >= CUTi:
        hits[0] = 1
    
        plcomp = compare_residues( sc1, sc2 )

        if int(plcomp[4]) >= CUTi:
            hits[1] = 1

            if abs(plcomp[0]) >= 0.8:#planar dot product between sidechains
                hits[2] = 1

    return hits
