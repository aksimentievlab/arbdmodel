#!/usr/bin/python

import string
from sys import argv,stdout
from os import popen,system
from os.path import exists,basename
from math import log, sqrt
from numpy import cross, array, dot, vdot, arccos
import numpy as np
import random

import elife_planegeo as pl
from elife_planegeo import rgroups, agroups

#Dictionary of SP2 sidechains
comp = {}
comp["ASN"] = ["CB", "CG", "OD1", "ND2"]
comp["GLN"] = ["CG", "CD", "OE1", "NE2"]
comp["ASP"] = ["CB", "CG", "OD1", "OD2"]
comp["GLU"] = ["CG", "CD", "OE1", "OE2"]
comp["ARG"] = ["NE", "CZ", "NH1", "NH2"]
comp["HIS"] = ["CB", "CG", "ND1", "CD2", "CE1", "NE2"]
comp["PHE"] = ["CB","CG","CD1","CD2","CE1","CE2","CZ"]
comp["TYR"] = ["CB","CG","CD1","CD2","CE1","CE2","CZ"]
comp["TRP"] = ["CB", "CG", "CD2", "CD1", "NE1", "CE2", "CZ2", "CH2", "CZ3", "CE3"]

if len(argv) == 1:
    print "USAGE:"
    print
    print "   ./elife_write_PDBcontacts.py [pdb file]          -> All protein chains "
    print "   ./elife_write_PDBcontacts.py [pdb file] [chain]  -> Specified chain "
    print
    print "ANNOTATION FORMAT:"
    print
    print "   A.143.HIS*A.159.PHE         -> Contact between two sidechains (chain A, i. 143 and i. 159)"
    print
    print "   A.98.TYR*A.99.VALTHR        -> Contact between sidechain at chain A and i. 98 to backbone"
    print "                                  peptide bond between i. 99 and i. 100"
    print
    print "   A.208.ASPPRO*A.206.GLUALA   -> Planar contact between two backbone peptide bonds"
    print
    print "   TOT.226 # CYS.1 MET.6 [...] -> Residue count numbers for observed amino acid types"
    print
    exit()
    
pdbf = argv[1]

ARGVCHAIN = ""
if len(argv) == 3: ARGVCHAIN = argv[2]

pdbdict = pl.read_pdb(pdbf)
pl.is_complete( pdbdict )

condict = {}
aadict = {}

for x in pdbdict.keys():
    aa1 = pdbdict[x][0]
    res1 = int(x.split('.')[1])
    chain1 = x.split('.')[0]
    if pdbdict[x][2] and (chain1 == ARGVCHAIN or ARGVCHAIN == ""):

        res1_next = int(x.split('.')[1])+1
        xnext = chain1+'.'+str(res1_next)
        
        
        if aadict.has_key(chain1) == False:
            aadict[chain1] = {}
            condict[chain1] = {}

        if aadict[chain1].has_key(aa1) == False:
            aadict[chain1][aa1] = 0

        aadict[chain1][aa1] += 1

        for y in pdbdict.keys():

            aa2 = pdbdict[y][0]
            res2 = int(y.split('.')[1])
            chain2 = y.split('.')[0]
            
            if pdbdict[y][2]:

                CADIST = pl.CA_distance(pdbdict[x], pdbdict[y])
                
                if CADIST < 24.0:

                    if x != y and comp.has_key(pdbdict[x][0]) \
                       and comp.has_key(pdbdict[y][0]) and CADIST > 1.0: 

                        stataa = pl.allcompare_sc2sc( pdbdict[x], pdbdict[y])

                        if stataa[2] == 1:
                            cname  = x+"."+aa1+"*"+y+"."+aa2
                            cnamer = y+"."+aa2+"*"+x+"."+aa1
                            if condict[chain1].has_key(cname) == False and \
                               condict[chain1].has_key(cnamer) == False:
                                print cname
                                condict[chain1][cname] = 0
                                condict[chain1][cnamer] = 0

                    res2_next = int(y.split('.')[1])+1
                    ynext = chain2+'.'+str(res2_next)

                    if pdbdict.has_key(ynext) and comp.has_key(pdbdict[x][0]):

                        statbb = pl.allcompare_bb2sc( pdbdict[y], pdbdict[ynext], pdbdict[x] )

                        if statbb[2] == 1:
                            aan = pdbdict[ynext][0]
                            cname = x+"."+aa1+"*"+y+"."+aa2+aan
                            cnamer = y+"."+aa2+"*"+x+"."+aa1

                            if condict[chain1].has_key(cname) == False and \
                               condict[chain1].has_key(cnamer) == False:
                                print cname
                                condict[chain1][cname] = 0
                                condict[chain1][cnamer] = 0

                    if pdbdict.has_key(xnext) and pdbdict.has_key(ynext) and \
                       ( abs(res1-res2) > 0 or chain1 != chain2 ):
                        if pdbdict[xnext][2]:

                            statbb = pl.allcompare_bb2bb( pdbdict[y], pdbdict[ynext], pdbdict[x], pdbdict[xnext] )

                            if statbb[2] == 1:
                                aaxn = pdbdict[xnext][0]
                                aayn = pdbdict[ynext][0]
                                cname = x+"."+aa1+aaxn+"*"+y+"."+aa2+aayn
                                cnamer = y+"."+aa2+aayn+"*"+x+"."+aa1+aaxn
                                
                                if condict[chain1].has_key(cname) == False and \
                               condict[chain1].has_key(cnamer) == False:
                                    print cname
                                    condict[chain1][cname] = 0
                                    condict[chain1][cnamer] = 0

for chain in aadict.keys():
    if (chain == ARGVCHAIN or ARGVCHAIN == ""):

        if ARGVCHAIN == "":
            print "CHAIN:", chain,

        TOT = 0
        for AA in aadict[chain].keys():
            TOT += aadict[chain][AA]
        print "TOT."+str(TOT), "#",
        for AA in aadict[chain].keys():
            print AA+"."+str(aadict[chain][AA]),
        print
