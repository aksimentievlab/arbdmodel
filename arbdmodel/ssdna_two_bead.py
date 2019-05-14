# -*- coding: utf-8 -*-
import numpy as np
import sys

## Local imports
from . import ArbdModel, ParticleType, PointParticle, Group, get_resource_path
from .abstract_polymer import PolymerSection, AbstractPolymerGroup
from .interactions import TabulatedPotential, HarmonicBond, HarmonicAngle, HarmonicDihedral
from .coords import quaternion_to_matrix

"""Define particle types"""

## units "295 k K/(160 amu * 1.24/ps)" "AA**2/ns"
## units "295 k K/(180 amu * 1.24/ps)" "AA**2/ns"
_P = ParticleType("P",
                 diffusivity = 1621,
                 mass = 121,
                 radius = 5,
                 nts = 0.5      # made compatible with nbPot
)

_B = ParticleType("B",
                 diffusivity = 1093,
                 mass = 181,    # thymine
                 radius = 3,                 
                 nts = 0.5      # made compatible with nbPot
)

class DnaStrandFromPolymer(Group):
    p = PointParticle(_P, (0,0,0), "P")
    b = PointParticle(_B, (3,0,1), "B")
    nt = Group( name = "nt", children = [p,b])
    nt.add_bond( i=p, j=b, bond = get_resource_path('two_bead_model/BPB.dat'), exclude = True )

    def __init__(self, polymer, **kwargs):
        self.polymer = polymer
        Group.__init__(self, **kwargs)
        
    def _clear_beads(self):
        ...
        
    def _generate_beads(self):
        nts = self.nts = self.children

        for i in range(self.polymer.num_monomers):
            c = self.polymer.monomer_index_to_contour(i)
            r = self.polymer.contour_to_position(c)
            o = self.polymer.contour_to_orientation(c)
            
            new = DnaStrandFromPolymer.nt.duplicate()
            new.orientation = o
            new.position = r
            self.add(new)

        ## Two consecutive nts 
        for i in range(len(nts)-1):
            p1,b1 = nts[i].children
            p2,b2 = nts[i+1].children
            self.add_bond( i=b1, j=p2, bond = get_resource_path('two_bead_model/BBP.dat'), exclude=True )
            self.add_bond( i=p1, j=p2, bond = get_resource_path('two_bead_model/BPP.dat'), exclude=True )
            self.add_angle( i=p1, j=p2, k=b2, angle = get_resource_path('two_bead_model/p1p2b2.dat') )
            self.add_angle( i=b1, j=p2, k=b2, angle = get_resource_path('two_bead_model/b1p2b2.dat') )
            self.add_dihedral( i=b1, j=p1, k=p2, l=b2, dihedral = get_resource_path('two_bead_model/b1p1p2b2.dat') )
            self.add_exclusion( i=b1, j=b2 )
            self.add_exclusion( i=p1, j=b2 )

        ## Three consecutive nts 
        for i in range(len(nts)-2):
            p1,b1 = nts[i].children
            p2,b2 = nts[i+1].children
            p3,b3 = nts[i+2].children
            self.add_angle( i=p1, j=p2, k=p3, angle = get_resource_path('two_bead_model/p1p2p3.dat') )
            self.add_angle( i=b1, j=p2, k=p3, angle = get_resource_path('two_bead_model/b1p2p3.dat') )
            self.add_dihedral( i=b1, j=p2, k=p3, l=b3, dihedral = get_resource_path('two_bead_model/b1p2p3b3.dat') )
            self.add_exclusion( i=p1, j=p3 )
            self.add_exclusion( i=b1, j=p3 )

        ## Four consecutive nts 
        for i in range(len(nts)-4):
            p1,b1 = nts[i].children
            p2,b2 = nts[i+1].children
            p3,b3 = nts[i+2].children
            p4,b4 = nts[i+3].children
            self.add_dihedral( i=p1, j=p2, k=p3, l=p4, dihedral = get_resource_path('two_bead_model/p0p1p2p3.dat') )

# def hybridize(strand1, strand2, parent=None, num_bp=None, start1=None, end2=None): 

#     """ hybridize num_bp basepairs between strand1 and strand2,
#     starting with nt at start1 and ending with nucleotide and end2 """

#     num_nt1 = len(strand1.children)
#     num_nt2 = len(strand2.children)

#     if parent is None:
#         assert(strand1.parent == strand2.parent)
#         parent = strand1.parent

#     if start1 is None:
#         start1 = 0
#     if end2 is None:
#         end2 = num_nt2

#     assert(start1 >= 0)
#     assert(end2 > 0)
#     assert(start1 < num_nt1)
#     assert(end2 <= num_nt2)
    
#     if num_bp is None:
#         num_bp = min(num_nt1-start1, end2)
        
#     if num_bp > num_nt1-start1:
#         raise ValueError("Attempted to hybridize too many basepairs ({}) for strand1 ({})".format(num_bp,num_nt1-start1))
#     if num_bp > end2:
#         raise ValueError("Attempted to hybridize too many basepairs ({}) for strand1 ({})".format(num_bp,end2))

#     nts1 = strand1.children[start1:start1+num_bp]
#     nts2 = strand2.children[end2-num_bp:end2][::-1]
#     assert( len(nts1) == len(nts2) )

#     kAngle = 0.0274155          # 90 kcal_mol per radian**2 

#     ## every bp
#     for i in range(num_bp):
#         p11,b11 = nts1[i].children
#         p21,b21 = nts2[i].children
#         parent.add_bond( i=b11, j=b21, bond=HarmonicBond(10,7.835) )
#         parent.add_angle( i=p11, j=b11, k=b21, angle=HarmonicAngle(kAngle,162.0) )
#         parent.add_angle( i=p21, j=b21, k=b11, angle=HarmonicAngle(kAngle,162.0) )

#     ## every 2 bp
#     for i in range(num_bp-1):
#         p11,b11 = nts1[i].children
#         p12,b12 = nts1[i+1].children
#         p21,b21 = nts2[i].children
#         p22,b22 = nts2[i+1].children

#         parent.add_bond( i=b11, j=b22, bond=HarmonicBond(1,8.41) )
#         parent.add_bond( i=b12, j=b21, bond=HarmonicBond(1,7.96) )
#         parent.add_angle( i=p11, j=p12, k=b12, angle=HarmonicAngle(kAngle,87.0) )
#         parent.add_angle( i=p22, j=p21, k=b11, angle=HarmonicAngle(kAngle,87.0) )

#     ## every 3 bp
#     for i in range(num_bp-2):
#         p11,b11 = nts1[i].children
#         p12,b12 = nts1[i+1].children
#         p13,b13 = nts1[i+2].children
#         p21,b21 = nts2[i].children
#         p22,b22 = nts2[i+1].children
#         p23,b23 = nts2[i+2].children

#         parent.add_angle( i=p11, j=p12, k=p13, angle=HarmonicAngle(kAngle,150.0) )
#         parent.add_angle( i=p21, j=p22, k=p23, angle=HarmonicAngle(kAngle,150.0) )

class DnaModel(ArbdModel):
    def __init__(self, polymers,
                 DEBUG=False,
                 **kwargs):

        kwargs['timestep'] = 20e-6
        kwargs['cutoff'] = 35
        
        self.polymer_group = AbstractPolymerGroup(polymers)
        self.strands = [DnaStrandFromPolymer(p) for p in self.polymer_group.polymers]

        ArbdModel.__init__(self, self.strands, **kwargs)

        self.useNonbondedScheme( TabulatedPotential(get_resource_path('two_bead_model/NBBB.dat')), typeA=_B, typeB=_B )
        self.useNonbondedScheme( TabulatedPotential(get_resource_path('two_bead_model/NBPB.dat')), typeA=_P, typeB=_B )
        self.useNonbondedScheme( TabulatedPotential(get_resource_path('two_bead_model/NBPP.dat')), typeA=_P, typeB=_P )

        self.generate_beads()
                
    def generate_beads(self):
        for s in self.strands:
            s._generate_beads()
        


if __name__ == "__main__":
    strands = []
    for i in range(5,60,5):

        strands.extend( [strand.duplicate() for j in range(int(round(600/i**1.2)))] )

    ## Randomly place strands through system
    model = ArbdModel( strands, dimensions=(200, 200, 200) )

    old_schemes = model.nbSchemes
    model.nbSchemes = []
    model.useNonbondedScheme( TabulatedPotential(get_resource_path('two_bead_model/NBBB.dat')), typeA=B, typeB=B )
    model.useNonbondedScheme( TabulatedPotential(get_resource_path('two_bead_model/NBPB.dat')), typeA=P, typeB=B )
    model.useNonbondedScheme( TabulatedPotential(get_resource_path('two_bead_model/NBPP.dat')), typeA=P, typeB=P )
    model.nbSchemes.extend(old_schemes)

    for s in strands:
        s.orientation = randomOrientation()
        s.position = np.array( [(a-0.5)*b for a,b in 
                                zip( np.random.uniform(size=3), model.dimensions )] )

    model.simulate( output_name = 'many-strands', output_period=1e4, num_steps=1e6 )
