# -*- coding: utf-8 -*-
## Test with `python -m arbdmodel.hps_polymer_model`

import numpy as np
import sys


## Local imports
from . import logger, ParticleType, PointParticle
from .polymer import PolymerBeads, PolymerModel
from .interactions import AbstractPotential, HarmonicBond

"""Define particle types"""
_types = dict(
    A = ParticleType("ALA",
                     mass = 71.08,
                     charge = 0,
                     sigma = 5.04,
                     lambda_ = 0.72973,
                 ),
    R = ParticleType("ARG",
                     mass = 156.2,
                     charge = 1,
                     sigma = 6.56,
                     lambda_ = 0.0,
                 ),
    N = ParticleType("ASN",
                     mass = 114.1,
                     charge = 0,
                     sigma = 5.68,
                     lambda_ = 0.432432,
                 ),
    D = ParticleType("ASP",
                     mass = 115.1,
                     charge = -1,
                     sigma = 5.58,
                     lambda_ = 0.378378,
                 ),
    C = ParticleType("CYS",
                     mass = 103.1,
                     charge = 0,
                     sigma = 5.48,
                     lambda_ = 0.594595,
                 ),
    Q = ParticleType("GLN",
                     mass = 128.1,
                     charge = 0,
                     sigma = 6.02,
                     lambda_ = 0.513514,
                 ),
    E = ParticleType("GLU",
                     mass = 129.1,
                     charge = -1,
                     sigma = 5.92,
                     lambda_ = 0.459459,
                 ),
    G = ParticleType("GLY",
                     mass = 57.05,
                     charge = 0,
                     sigma = 4.5,
                     lambda_ = 0.648649,
                 ),
    H = ParticleType("HIS",
                     mass = 137.1,
                     charge = 0.5,
                     sigma = 6.08,
                     lambda_ = 0.513514,
                 ),
    I = ParticleType("ILE",
                     mass = 113.2,
                     charge = 0,
                     sigma = 6.18,
                     lambda_ = 0.972973,
                 ),
    L = ParticleType("LEU",
                     mass = 113.2,
                     charge = 0,
                     sigma = 6.18,
                     lambda_ = 0.972973,
                 ),
    K = ParticleType("LYS",
                     mass = 128.2,
                     charge = 1,
                     sigma = 6.36,
                     lambda_ = 0.513514,
                 ),
    M = ParticleType("MET",
                     mass = 131.2,
                     charge = 0,
                     sigma = 6.18,
                     lambda_ = 0.837838,
                 ),
    F = ParticleType("PHE",
                     mass = 147.2,
                     charge = 0,
                     sigma = 6.36,
                     lambda_ = 1.0,
                 ),
    P = ParticleType("PRO",
                     mass = 97.12,
                     charge = 0,
                     sigma = 5.56,
                     lambda_ = 1.0,
                 ),
    S = ParticleType("SER",
                     mass = 87.08,
                     charge = 0,
                     sigma = 5.18,
                     lambda_ = 0.594595,
                 ),
    T = ParticleType("THR",
                     mass = 101.1,
                     charge = 0,
                     sigma = 5.62,
                     lambda_ = 0.675676,
                 ),
    W = ParticleType("TRP",
                     mass = 186.2,
                     charge = 0,
                     sigma = 6.78,
                     lambda_ = 0.945946,
                 ),
    Y = ParticleType("TYR",
                     mass = 163.2,
                     charge = 0,
                     sigma = 6.46,
                     lambda_ = 0.864865,
                 ),
    V = ParticleType("VAL",
                     mass = 99.07,
                     charge = 0,
                     sigma = 5.86,
                     lambda_ = 0.891892,
                 )
)
for k,t in _types.items():
    t.resname = t.name

class HpsNonbonded(AbstractPotential):
    def __init__(self, debye_length=10, resolution=0.1, range_=(0,None)):
        AbstractPotential.__init__(self, resolution=resolution, range_=range_)
        self.debye_length = debye_length
        self.max_force = 50

    def potential(self, r, types):
        """ Electrostatics """
        typeA,typeB = types
        ld = self.debye_length 
        q1 = typeA.charge
        q2 = typeB.charge
        D = 80                  # dielectric of water
        ## units "e**2 / (4 * pi * epsilon0 AA)" kcal_mol
        A =  332.06371
        u_elec = (A*q1*q2/D)*np.exp(-r/ld) / r 
        
        """ Hydrophobicity scale model """
        lambda_ = 0.5 * (typeA.lambda_ + typeB.lambda_)
        sigma = 0.5 * (typeA.sigma + typeB.sigma)
        epsilon = 0.2
        
        r6 = (sigma/r)**6
        r12 = r6**2
        u_lj = 4 * epsilon * (r12-r6)
        u_hps = lambda_ * np.array(u_lj)
        s = r<=sigma*2**(1/6)
        u_hps[s] = u_lj[s] + (1-lambda_) * epsilon

        u = u_elec + u_hps
        return u

class HpsBeads(PolymerBeads):

    def __init__(self, polymer, sequence=None,
                 spring_constant = 2.3900574,
                 rest_length = 3.8, **kwargs):

        if sequence is None:
            raise NotImplementedError
            # ... set random sequence

        self.spring_constant = spring_constant
        PolymerBeads.__init__(self, polymer, sequence, rest_length=rest_length, **kwargs)

        assert(self.monomers_per_bead_group == 1)
        
        if len(sequence) != polymer.num_monomers:
            raise ValueError("Length of sequence does not match length of polymer")                

    def _generate_ith_bead_group(self, i, r, o):
        s = self.sequence[i]
        return PointParticle(_types[s], r,
                             name = s,
                             resid = i+1)

    def _join_adjacent_bead_groups(self, ids):

        ## Two consecutive nts 
        if len(ids) == 2:
            b1,b2 = [self.children[i] for i in ids]
            """ units "10 kJ/N_A" kcal_mol """
            bond = HarmonicBond(k = self.spring_constant,
                                r0 = self.rest_length,
                                range_ = (0,500),
                                resolution = 0.01,
                                max_force = 10)

            self.add_bond( i=b1, j=b2, bond = bond, exclude=True )
        elif len(ids) == 3:
            ...
        else:
            pass

class HpsModel(PolymerModel):
    def __init__(self, polymers,
                 sequences = None,
                 rest_length = 3.8,
                 spring_constant = 2.3900574,
                 debye_length = 10,
                 damping_coefficient = 10,
                 DEBUG=False,
                 **kwargs):

        """ 
        [debye_length]: angstroms
        [damping_coefficient]: 1/ns
        """
        if 'timestep' not in kwargs: kwargs['timestep'] = 10e-6
        if 'cutoff' not in kwargs: kwargs['cutoff'] = max(4*debye_length,20)

        if 'decomp_period' not in kwargs:
            kwargs['decomp_period'] = 1000

        self.rest_length = rest_length
        self.spring_constant = spring_constant

        """ Assign sequences """
        if sequences is None:
            raise NotImplementedError("HpsModel must be provided a sequences argument")

        PolymerModel.__init__(self, polymers, sequences, monomers_per_bead_group=1, **kwargs)

        """ Update type diffusion coefficients """
        self.types = all_types = [t for key,t in _types.items()]
        self.set_damping_coefficient( damping_coefficient )

        """ Set up nonbonded interactions """
        nonbonded = HpsNonbonded(debye_length)
        self.types
        for i in range(len(all_types)):
            t1 = all_types[i]
            for j in range(i,len(all_types)):
                t2 = all_types[j]
                self.add_nonbonded_interaction( nonbonded, typeA=t1, typeB=t2 )
                
    def _generate_polymer_beads(self, polymer, sequence, polymer_index=None):
        return HpsBeads(polymer, sequence,
                        rest_length = self.rest_length,
                        spring_constant = self.spring_constant,
                        monomers_per_bead_group = self.monomers_per_bead_group,
                        polymer_index = polymer_index
                        )

    def set_damping_coefficient(self, damping_coefficient):
        for t in self.types:
            t.damping_coefficient = damping_coefficient
            # t.diffusivity = 831447.2 * temperature / (t.mass * damping_coefficient)

if __name__ == "__main__":

    print("TYPES")
    for n,t in _types.items():
        print("{}\t{}\t{}\t{}\t{}".format(t.name, t.mass, t.charge, t.sigma, t.lambda_))
