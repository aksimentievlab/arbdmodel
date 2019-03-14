# -*- coding: utf-8 -*-
## Test with `python -m arbdmodel.hps_polymer_model`

import numpy as np
import sys


## Local imports
from . import ArbdModel, ParticleType, PointParticle, Group, get_resource_path    
from .abstract_polymer import PolymerSection, AbstractPolymerGroup
from .interactions import NonbondedScheme, HarmonicBond, HarmonicAngle, HarmonicDihedral
from .coords import quaternion_to_matrix

"""Define particle types"""
_types = dict(
    A = ParticleType("ALA",
                     mass = 71.08,
                     charge = 0,
                     sigma = 5.04,
                     lambda_ = 0.730,
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
                     lambda_ = 0.432,
                 ),
    D = ParticleType("ASP",
                     mass = 115.1,
                     charge = -1,
                     sigma = 5.58,
                     lambda_ = 0.378,
                 ),
    C = ParticleType("CYS",
                     mass = 103.1,
                     charge = 0,
                     sigma = 5.48,
                     lambda_ = 0.595,
                 ),
    Q = ParticleType("GLN",
                     mass = 128.1,
                     charge = 0,
                     sigma = 6.02,
                     lambda_ = 0.514,
                 ),
    E = ParticleType("GLU",
                     mass = 129.1,
                     charge = -1,
                     sigma = 5.92,
                     lambda_ = 0.459,
                 ),
    G = ParticleType("GLY",
                     mass = 57.05,
                     charge = 0,
                     sigma = 4.5,
                     lambda_ = 0.649,
                 ),
    H = ParticleType("HIS",
                     mass = 137.1,
                     charge = 0.5,
                     sigma = 6.08,
                     lambda_ = 0.514,
                 ),
    I = ParticleType("ILE",
                     mass = 113.2,
                     charge = 0,
                     sigma = 6.18,
                     lambda_ = 0.973,
                 ),
    L = ParticleType("LUE",
                     mass = 113.2,
                     charge = 0,
                     sigma = 6.18,
                     lambda_ = 0.973,
                 ),
    K = ParticleType("LYS",
                     mass = 128.2,
                     charge = 1,
                     sigma = 6.36,
                     lambda_ = 0.514,
                 ),
    M = ParticleType("MET",
                     mass = 131.2,
                     charge = 0,
                     sigma = 6.18,
                     lambda_ = 0.838,
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
                     lambda_ = 0.595,
                 ),
    T = ParticleType("THR",
                     mass = 101.1,
                     charge = 0,
                     sigma = 5.62,
                     lambda_ = 0.676,
                 ),
    W = ParticleType("TRP",
                     mass = 186.2,
                     charge = 0,
                     sigma = 6.78,
                     lambda_ = 0.946,
                 ),
    Y = ParticleType("TYR",
                     mass = 163.2,
                     charge = 0,
                     sigma = 6.46,
                     lambda_ = 0.865,
                 ),
    V = ParticleType("VAL",
                     mass = 99.07,
                     charge = 0,
                     sigma = 5.86,
                     lambda_ = 0.892,
                 )
)

class HpsNonbonded(NonbondedScheme):
    def __init__(self, debye_length=10, resolution=0.1, rMin=0):
        NonbondedScheme.__init__(self, typesA=None, typesB=None, resolution=resolution, rMin=rMin)
        self.debye_length = debye_length
        self.maxForce = 50

    def potential(self, r, typeA, typeB):
        """ Electrostatics """
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
        u[0] = u[1]             # Remove NaN

        maxForce = self.maxForce
        if maxForce is not None:
            assert(maxForce > 0)
            f = np.diff(u)/np.diff(r)
            f[f>maxForce] = maxForce
            f[f<-maxForce] = -maxForce
            u[0] = 0
            u[1:] = np.cumsum(f*np.diff(r))
        
        u = u-u[-1]
            
        return u

class HpsBeadsFromPolymer(Group):
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
        if len(sequence) != polymer.num_monomers:
            raise ValueError("Length of sequence does not match length of polymer")
        Group.__init__(self, **kwargs)
        
    def _clear_beads(self):
        ...
        
    def _generate_beads(self):
        # beads = self.children

        for i in range(self.polymer.num_monomers):
            c = self.polymer.monomer_index_to_contour(i)
            r = self.polymer.contour_to_position(c)
            s = self.sequence[i]

            bead = PointParticle(_types[s], r )
            self.add(bead)

        ## Two consecutive nts 
        for i in range(len(self.children)-1):
            b1 = self.children[i]
            b2 = self.children[i+1]
            """ units "10 kJ/N_A" kcal_mol """
            bond = HarmonicBond(k = 2.3900574,
                                r0 = 3.8,
                                rRange = (0,500),
                                resolution = 0.01,
                                maxForce = 10)
            self.add_bond( i=b1, j=b2, bond = bond, exclude=True )


class HpsModel(ArbdModel):
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

        kwargs['timestep'] = 10e-6
        kwargs['cutoff'] = max(4*debye_length,20)

        """ Assign sequences """
        if sequences is None:
            raise NotImplementedError("HpsModel must be provided a sequences argument")

        self.polymer_group = AbstractPolymerGroup(polymers)
        self.peptides = [HpsBeadsFromPolymer(p,s) 
                         for p,s in zip(self.polymer_group.polymers,sequences)]

        ArbdModel.__init__(self, self.peptides, **kwargs)

        """ Update type diffusion coefficients """
        all_types = [t for key,t in _types.items()]
        for t in all_types:
            t.damping_coefficient = damping_coefficient
            # t.diffusivity = 831447.2 * temperature / (t.mass * damping_coefficient)

        """ Set up nonbonded interactions """
        nonbonded = HpsNonbonded(debye_length)
        for i in range(len(all_types)):
            t1 = all_types[i]
            for j in range(i,len(all_types)):
                t2 = all_types[j]
                self.useNonbondedScheme( nonbonded, typeA=t1, typeB=t2 )
                
        """ Generate beads """
        self.generate_beads()

    def update_splines(self, coords):
        ...
                
    def generate_beads(self):
        for s in self.peptides:
            s._generate_beads()


if __name__ == "__main__":

    print("TYPES")
    for n,t in _types.items():
        print("{}\t{}\t{}\t{}\t{}".format(t.name, t.mass, t.charge, t.sigma, t.lambda_))
