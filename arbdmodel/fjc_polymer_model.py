# -*- coding: utf-8 -*-
## Test with `python -m arbdmodel.fjc_polymer_model`

import numpy as np
import sys

## Local imports
from . import logger, ParticleType, PointParticle
from .polymer import PolymerBeads, PolymerModel
from .interactions import AbstractPotential, HarmonicBond


"""Define particle types"""
type_ = ParticleType("X",
                     damping_coefficient = 40.9,
                     mass=120
)

# ## Bonded potentials
# class FjcNonbonded(AbstractPotential):
#     """ This potential should apply zero force; however, it is required for the arbd engine """
#     def __init__(self, resolution=0.1, range_=(0,1)):
#         AbstractPotential.__init__(self, resolution=resolution, range_=range_)

#     def potential(self, r, types):
#         u = np.zeros(r.shape)
#         return u

class FjcBeadsFromPolymer(PolymerBeads):

    def __init__(self, polymer, sequence=None, 
                 rest_length = None, spring_constant = 25, monomers_per_bead_group = 1,
                 **kwargs):

        self.spring_constant = spring_constant
        PolymerBeads.__init__(self, polymer, sequence, monomers_per_bead_group, rest_length, **kwargs)
        
    def _generate_ith_bead_group(self, i, r, o):
        return PointParticle(type_, r,
                             resid = i+1)

    def _join_adjacent_bead_groups(self, ids):
        if len(ids) == 2:
            b1,b2 = [self.children[i] for i in ids]
            """ units "10 kJ/N_A" kcal_mol """
            bond = HarmonicBond(k = self.spring_constant,
                                r0 = self.rest_length,
                                range_ = (0,500),
                                resolution = 0.01,
                                max_force = 50)
            # logger.info(f'Adding bond to {b1} {b2} with k={self.spring_constant} and r0={self.rest_length}')
            self.add_bond( i=b1, j=b2, bond = bond, exclude=True )
        else:
            pass

class FjcModel(PolymerModel):
    def __init__(self, polymers,
                 sequences = None,
                 rest_length = None,
                 monomers_per_bead_group = 1,
                 spring_constant = 25,
                 damping_coefficient = 40.9,
                 DEBUG=False,
                 **kwargs):

        """ 
        [damping_coefficient]: ns
        """
        
        if 'timestep' not in kwargs: kwargs['timestep'] = 50e-6
        if 'cutoff' not in kwargs: kwargs['cutoff'] = 5 # no non-bonded interactions, so set this small
        if 'decomp_period' not in kwargs:
            kwargs['decomp_period'] = 100000 # Make it huge because we never need to compute FJC NB forces

        self.rest_length = rest_length
        self.spring_constant = spring_constant
        PolymerModel.__init__(self, polymers, sequences, monomers_per_bead_group, **kwargs)

        """ Update type diffusion coefficients """
        logger.warning("Diffusion coefficient arbitrarily set to 100 AA**2/ns in FjcModel")
        self.set_damping_coefficient( damping_coefficient )

        """ Set up nonbonded interactions """
        # nonbonded = FjcNonbonded()
        #  self.add_nonbonded_interaction( nonbonded, typeA=type_, typeB=type_ )

    def _generate_polymer_beads(self, polymer, sequence):
        return FjcBeadsFromPolymer(polymer, sequence,
                                   rest_length = self.rest_length,
                                   spring_constant = self.spring_constant,
                                   monomers_per_bead_group = self.monomers_per_bead_group,
                                   )

    def set_damping_coefficient(self, damping_coefficient):
        for t in [type_]:
            t.damping_coefficient = damping_coefficient
            # t.diffusivity = 831447.2 * temperature / (t.mass * damping_coefficient)

if __name__ == "__main__":
    pass
