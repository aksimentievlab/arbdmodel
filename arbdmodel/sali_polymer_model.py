# -*- coding: utf-8 -*-
## Test with `python -m arbdmodel.hps_polymer_model`

import numpy as np
import sys


## Local imports
from . import ParticleType, PointParticle, Group
from .logger import logger, get_resource_path    
from .model import ArbdModel
from .polymer import PolymerSection, PolymerGroup
from .interactions import AbstractPotential, HarmonicBond, HarmonicBondedPotential
from .coords import quaternion_to_matrix


"""Define particle types"""
type_ = ParticleType("AA",
                     # units "297.15 k K / (6 pi 0.92 mPa s 6 AA)" "AA**2/ns"
                     diffusivity = 39.429314
)

## Bonded potentials
class LinearBond(HarmonicBond):
    def __init__(self, k, r0, rRange=(0,50), resolution=0.1, maxForce=None, max_potential=None, prefix="potentials/"):
        HarmonicBondedPotential.__init__(self, k, r0, rRange, resolution, maxForce, max_potential, prefix)
        self.type_ = "linearbond"
        self.kscale_ = 1.0

    def potential(self, dr):
        return self.k*np.abs(dr)

class SaliNonbonded(AbstractPotential):
    def __init__(self, resolution=0.1):
        AbstractPotential.__init__(self, resolution=resolution)

    def potential(self, r):
        """ Constant force excluded volume """
        force = 10              # kcal_mol/AA
        radius = 6
        
        u = np.zeros(r.shape)
        s = r < 2*radius
        u[s] = (2*radius - r[s]) * force            
        return u

class SaliBeadsFromPolymer(Group):
    # p = PointParticle(_P, (0,0,0), "P")
    # b = PointParticle(_B, (3,0,1), "B")
    # nt = Group( name = "nt", children = [p,b])
    # nt.add_bond( i=p, j=b, bond = get_resource_path('two_bead_model/BPB.dat') )

    def __init__(self, polymer, sequence=None, **kwargs):

        if sequence is None:
            raise NotImplementedError
            # ... set random sequence

        self.polymer = polymer
        self.sequence = sequence

        for prop in ('segname','chain'):
            if prop not in kwargs:
                # import pdb
                # pdb.set_trace()
                try:
                    self.__dict__[prop] = polymer.__dict__[prop]
                except:
                    pass

        if len(sequence) != polymer.num_monomers:
            raise ValueError("Length of sequence does not match length of polymer")
        Group.__init__(self, **kwargs)
        
        self.num_beads = int(np.ceil(polymer.num_monomers / 20))

    def _clear_beads(self):
        ...
        
    def _generate_beads(self):
        # beads = self.children
        nb = self.num_beads
        
        for i in range(nb):
            # c = self.polymer.monomer_index_to_contour(i)
            c = float(i)/(nb-1) if nb > 1 else 0.5
            r = self.polymer.contour_to_position(c)

            bead = PointParticle(type_, r,
                                 resid = i+1)
            self.add(bead)

        ## Two consecutive nts 
        for i in range(len(self.children)-1):
            b1 = self.children[i]
            b2 = self.children[i+1]
            """ units "10 kJ/N_A" kcal_mol """
            bond = LinearBond(k = 1,
                              r0 = 18,
                              rRange = (0,500),
                              resolution = 0.01,
                              maxForce = 30)
            self.add_bond( i=b1, j=b2, bond = bond, exclude=True )

class SaliModel(ArbdModel):
    def __init__(self, polymers,
                 sequences = None,
                 debye_length = 10,
                 damping_coefficient = 10,
                 DEBUG=False,
                 **kwargs):

        """ 
        [debye_length]: angstroms
        [damping_coefficient]: ns
        """

        print("WARNING: diffusion coefficient arbitrarily set to 100 AA**2/ns in SaliModel")
        
        kwargs['timestep'] = 1047e-6
        kwargs['cutoff'] = 18

        if 'decompPeriod' not in kwargs:
            kwargs['decompPeriod'] = 1000

        """ Assign sequences """
        if sequences is None:
            raise NotImplementedError("HpsModel must be provided a sequences argument")

        self.polymer_group = PolymerGroup(polymers)
        self.sequences = sequences
        ArbdModel.__init__(self, [], **kwargs)

        # """ Update type diffusion coefficients """
        # self.set_damping_coefficient( damping_coefficient )

        """ Set up nonbonded interactions """
        nonbonded = SaliNonbonded()
        self.useNonbondedScheme( nonbonded, typeA=type_, typeB=type_ )
                
        """ Generate beads """
        self.generate_beads()

    def update_splines(self, coords):
        i = 0
        for p,c in zip(self.polymer_group.polymers,self.children):
            n = len(c.children)
            p.set_splines(np.linspace(0,1,n), coords[i:i+n])
            i += n

        self.clear_all()
        self.generate_beads()
        ## TODO Apply restraints, etc

    def generate_beads(self):
        self.peptides = [SaliBeadsFromPolymer(p,s)
                         for p,s in zip(self.polymer_group.polymers,self.sequences)]

        for s in self.peptides:
            self.add(s)
            s._generate_beads()

    # def set_damping_coefficient(self, damping_coefficient):
    #     for t in self.types:
    #         t.damping_coefficient = damping_coefficient
    #         # t.diffusivity = 831447.2 * temperature / (t.mass * damping_coefficient)

if __name__ == "__main__":
    pass
