# ARBD modeling

Provides a Python interface to the ARBD simulation engine.

The interface currently supports the construction of simulation
systems out of point-like particles interacting through bonded and
non-bonded potentials. Rigid body particles are not yet supported.
The interface presently communicates with ARBD through files.

## Example: Constructing a coarse-grained ssDNA model

```python
import numpy as np
import sys
from arbdmodel import ArbdModel, ParticleType, PointParticle, Group
from arbdmodel.nonbonded import TabulatedPotential, HarmonicBond, HarmonicAngle, HarmonicDihedral

"""Define particle types"""
P = ParticleType("P",
                 diffusivity = 43.5,
                 mass = 150,
                 radius = 3,
                 nts = 0.5      # made compatible with nbPot
)

B = ParticleType("B",
                 diffusivity = 43.5,
                 mass = 150,
                 radius = 3,                 
                 nts = 0.5      # made compatible with nbPot
)

"""Function that creates a strand of DNA of a given length"""
def createDnaStrand(numNts):
    p = PointParticle(P, (0,0,0), "P")
    b = PointParticle(B, (3,0,1), "B")

    nt = Group( name = "nt", children = [p,b])
    nt.add_bond( i=p, j=b, bond = 'tabPot/BPB.dat' )
    nts = [nt.duplicate() for i in range( numNts )]

    strand = Group(name="strand", children=nts)

    for i in range(len(nts)):
        # nts[i].position[2] = i*4.5
        nts[i].children[0].position[2] = i*4.5
        nts[i].children[1].position[2] = i*4.5+1
        
    ## Two consecutive nts 
    for i in range(len(nts)-1):
        p1,b1 = nts[i].children
        p2,b2 = nts[i+1].children
        strand.add_bond( i=b1, j=p2, bond = 'tabPot/BBP.dat', exclude=True )
        strand.add_bond( i=p1, j=p2, bond = 'tabPot/BPP.dat', exclude=True )
        strand.add_angle( i=p1, j=p2, k=b2, angle = 'tabPot/p1p2b2.dat')
        strand.add_angle( i=b1, j=p2, k=b2, angle = 'tabPot/b1p2b2.dat')
        strand.add_dihedral( i=b1, j=p1, k=p2, l=b2, dihedral = 'tabPot/b1p1p2b2.dat')
        strand.add_exclusion( i=b1, j=b2 )
        strand.add_exclusion( i=p1, j=b2 )

    ## Three consecutive nts 
    for i in range(len(nts)-2):
        p1,b1 = nts[i].children
        p2,b2 = nts[i+1].children
        p3,b3 = nts[i+2].children
        strand.add_angle( i=p1, j=p2, k=p3, angle = 'tabPot/p1p2p3.dat')
        strand.add_angle( i=b1, j=p2, k=p3, angle = 'tabPot/b1p2p3.dat')
        strand.add_dihedral( i=b1, j=p2, k=p3, l=b3, dihedral = 'tabPot/b1p2p3b3.dat')
        strand.add_exclusion( i=p1, j=p3 )
        strand.add_exclusion( i=b1, j=p3 )
        
    ## Four consecutive nts 
    for i in range(len(nts)-4):
        p1,b1 = nts[i].children
        p2,b2 = nts[i+1].children
        p3,b3 = nts[i+2].children
        p4,b4 = nts[i+3].children
        strand.add_dihedral( i=p1, j=p2, k=p3, l=p4, dihedral = 'tabPot/p0p1p2p3.dat')

    return strand

if __name__ == "__main__":
    strands = []
    for i in range(5,60,5):
        strand = createDnaStrand(i)
        strands.extend( [strand.duplicate() for j in range(int(round(600/i**1.2)))] )

    ## Randomly place strands through system
	model = ArbdModel( strands, dimensions=(200, 200, 200) )

    old_schemes = model.nbSchemes
    model.nbSchemes = []
    model.useNonbondedScheme( TabulatedPotential('tabPot/NBBB.dat'), typeA=B, typeB=B )
    model.useNonbondedScheme( TabulatedPotential('tabPot/NBPB.dat'), typeA=P, typeB=B )
    model.useNonbondedScheme( TabulatedPotential('tabPot/NBPP.dat'), typeA=P, typeB=P )
    model.nbSchemes.extend(old_schemes)

    for s in strands:
        s.position = np.array( [(a-0.5)*b for a,b in 
                                zip( np.random.uniform(size=3), model.dimensions )] )

    model.simulate( output_name = 'many-strands', output_period=1e4, num_steps=1e6 )
```
