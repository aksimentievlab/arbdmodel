# -*- coding: utf-8 -*-
## Test with `python -m arbdmodel.hps_polymer_model`

import numpy as np
import sys


## Local imports
from . import ArbdModel, ParticleType, PointParticle, Group, get_resource_path    
from .polymer import PolymerSection, PolymerGroup
from .interactions import NonbondedScheme, HarmonicBond, HarmonicPotential
from .coords import quaternion_to_matrix


"""Define particle types"""
type_ = ParticleType("X",
                     damping_coefficient = 40.9,
                     mass=120
)

## Bonded potentials
class FjcNonbonded(NonbondedScheme):
    def __init__(self, resolution=0.1, rMin=0):
        NonbondedScheme.__init__(self, typesA=None, typesB=None, resolution=resolution, rMin=rMin)

    def potential(self, r, typeA, typeB):
        """ Constant force excluded volume """
        force = 10              # kcal_mol/AA
        radius = 6
        
        u = np.zeros(r.shape)
        # s = r < 2*radius
        # u[s] = (2*radius - r[s]) * force            
        return u

class FjcBeadsFromPolymer(Group):

    def __init__(self, polymer, sequence=None, 
                 rest_length = 3.8, spring_constant = 25,
                 **kwargs):

        # if sequence is None:
        #     raise NotImplementedError
        #     # ... set random sequence

        self.polymer = polymer
        self.sequence = sequence
        self.spring_constant = 25
        self.rest_length = 3.8

        for prop in ('segname','chain'):
            if prop not in kwargs:
                # import pdb
                # pdb.set_trace()
                try:
                    self.__dict__[prop] = polymer.__dict__[prop]
                except:
                    pass

        # if len(sequence) != polymer.num_monomers:
        #     raise ValueError("Length of sequence does not match length of polymer")
        Group.__init__(self, **kwargs)
        

    def _clear_beads(self):
        ...
        
    def _generate_beads(self):
        # beads = self.children
        nb = self.polymer.num_monomers
        
        for i in range(nb):
            c = self.polymer.monomer_index_to_contour(i)
            r = self.polymer.contour_to_position(c)

            bead = PointParticle(type_, r,
                                 resid = i+1)
            self.add(bead)

        ## Two consecutive nts 
        for i in range(len(self.children)-1):
            b1 = self.children[i]
            b2 = self.children[i+1]
            """ units "10 kJ/N_A" kcal_mol """
            bond = HarmonicBond(k = self.spring_constant,
                                r0 = self.rest_length,
                                rRange = (0,500),
                                resolution = 0.01,
                                maxForce = 50)
            self.add_bond( i=b1, j=b2, bond = bond, exclude=True )

class FjcModel(ArbdModel):
    def __init__(self, polymers,
                 sequences = None,
                 rest_length = 3.8,
                 spring_constant = 25,
                 damping_coefficient = 40.9,
                 DEBUG=False,
                 **kwargs):

        """ 
        [damping_coefficient]: ns
        """

        print("WARNING: diffusion coefficient arbitrarily set to 100 AA**2/ns in FjcModel")
        
        kwargs['timestep'] = 50e-6
        kwargs['cutoff'] = 10

        if 'decompPeriod' not in kwargs:
            kwargs['decompPeriod'] = 100000

        """ Assign sequences """
        if sequences is None:
            # raise NotImplementedError("HpsModel must be provided a sequences argument")
            sequences = [None for i in range(len(polymers))]

        self.polymer_group = PolymerGroup(polymers)
        self.sequences = sequences
        self.rest_length = rest_length
        self.spring_constant = spring_constant
        ArbdModel.__init__(self, [], **kwargs)

        """ Update type diffusion coefficients """
        self.set_damping_coefficient( damping_coefficient )

        """ Set up nonbonded interactions """
        nonbonded = FjcNonbonded()
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
        self.peptides = [FjcBeadsFromPolymer(p,s, rest_length = self.rest_length,
                                             spring_constant = self.spring_constant )
                         for p,s in zip(self.polymer_group.polymers,self.sequences)]

        for s in self.peptides:
            self.add(s)
            s._generate_beads()

    def set_damping_coefficient(self, damping_coefficient):
        for t in [type_]:
            t.damping_coefficient = damping_coefficient
            # t.diffusivity = 831447.2 * temperature / (t.mass * damping_coefficient)

if __name__ == "__main__":
    pass
