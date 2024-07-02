# -*- coding: utf-8 -*-
## Test with `python -m arbdmodel.mpipi_polymer_model`

import numpy as np
import sys
import pandas as pd
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
                     freq=0.091,
                 ),
    R = ParticleType("ARG",
                     mass = 156.2,
                     charge = 1,
                     sigma = 6.56,
                     freq=0.552
                 ),
    N = ParticleType("ASN",
                     mass = 114.1,
                     charge = 0,
                     sigma = 5.68,
                     freq=0.353
                 ),
    D = ParticleType("ASP",
                     mass = 115.1,
                     charge = -1,
                     sigma = 5.58,
                     freq=0.195,
                 ),
    C = ParticleType("CYS",
                     mass = 103.1,
                     charge = 0,
                     sigma = 5.48,
                     freq=0.127,
                 ),
    Q = ParticleType("GLN",
                     mass = 128.1,
                     charge = 0,
                     sigma = 6.02,
                     freq=0.365,
                 ),
    E = ParticleType("GLU",
                     mass = 129.1,
                     charge = -1,
                     sigma = 5.92,
                     freq=0.211,
                 ),
    G = ParticleType("GLY",
                     mass = 57.05,
                     charge = 0,
                     sigma = 4.5,
                     freq=0.22
                 ),
    H = ParticleType("HIS",
                     mass = 137.1,
                     charge = 0.5,
                     sigma = 6.08,
                     freq=0.668
                 ),
    I = ParticleType("ILE",
                     mass = 113.2,
                     charge = 0,
                     sigma = 6.18,
                     freq=0.005
                 ),
    L = ParticleType("LEU",
                     mass = 113.2,
                     charge = 0,
                     sigma = 6.18,
                     freq=0.021
                 ),
    K = ParticleType("LYS",
                     mass = 128.2,
                     charge = 1,
                     sigma = 6.36,
                     freq=0.048
                 ),
    M = ParticleType("MET",
                     mass = 131.2,
                     charge = 0,
                     sigma = 6.18,
                     freq=0.073
                 ),
    F = ParticleType("PHE",
                     mass = 147.2,
                     charge = 0,
                     sigma = 6.36,
                     freq=0.712
                 ),
    P = ParticleType("PRO",
                     mass = 97.12,
                     charge = 0,
                     sigma = 5.56,
                     freq=0.144
                 ),
    S = ParticleType("SER",
                     mass = 87.08,
                     charge = 0,
                     sigma = 5.18,
                     freq=0.113
                 ),
    T = ParticleType("THR",
                     mass = 101.1,
                     charge = 0,
                     sigma = 5.62,
                     freq=0.057,
                 ),
    W = ParticleType("TRP",
                     mass = 186.2,
                     charge = 0,
                     sigma = 6.78,
                     freq=1.0,
                 ),
    Y = ParticleType("TYR",
                     mass = 163.2,
                     charge = 0,
                     sigma = 6.46,
                     freq=0.762
                 ),
    V = ParticleType("VAL",
                     mass = 99.07,
                     charge = 0,
                     sigma = 5.86,
                     freq=0.011
                 )
)

for k,t in list(_types.items()):
    t.resname = t.name
    t.is_idp = False
    
    ## Add types for IDPs

    
class MpipiNonbonded(AbstractPotential):
    def __init__(self, debye_length=10, resolution=0.1, range_=(0,None)):
        AbstractPotential.__init__(self, resolution=resolution, range_=range_)
        self.debye_length = debye_length
        self.max_force = 50

    def potential(self, r, types):
        """ Electrostatics """
        typeA, typeB = types
        ld = self.debye_length 
        q1 = typeA.charge*0.75
        q2 = typeB.charge*0.75
        D = 80                  # dielectric of water
        ## units "e**2 / (4 * pi * epsilon0 AA)" kcal_mol
        A =  332.06371
        u_elec = (A*q1*q2/D)*np.exp(-r/ld) / r 

        """Mpipi Wang-Frenkel Potential"""
        WF=pd.read_csv("mpipi_protein_resname.csv",index_col=0)
        indices=list(df.columns)
        if indices.index(typeA.resname)<indices.index(typeB.resname):
            sigma_ij=WF[typeB.resname][typeA.resname+"_sigma"]*10
            eps_ij=WF[typeB.resname][typeA.resname+"_eps"]*0.23900574
        else:
            sigma_ij=WF[typeA.resname][typeB.resname+"_sigma"]*10
            eps_ij=WF[typeA.resname][typeB.resname+"_eps"]*0.23900574
        
        vij=1
        muij=2

        if typeA.resname=="ILE":
            if typeB.resname=="ILE":
                muij=11
            elif typeB.resname=="VAL"
                muij=4
        if typeB.resname=="ILE" and typeA.resname=="VAL": muij=4
            
        Rij=3*sigma_ij

        alpha=2*(3**(2*muij))*((2*vij+1/(2*vij*(3**(2*muij)-1))))**(2*vij+1)

        u_wang=eps_ij*alpha*((sigma_ij/r)**(2muij)-1)*((Rij/r)**2muij-1)**(2vij)

        """ Mpipi scale model """
        """
        A_is_idp = B_is_idp = False
        try:
            A_is_idp = typeA.is_idp
        except:
            pass
        try:
            B_is_idp = typeB.is_idp
        except:
            pass

        _idp_scale = (int(A_is_idp)*int(B_is_idp))

        alpha = 0.159 + _idp_scale * (0.228 - 0.159)
        epsilon0 = -1.36 + _idp_scale * (1.36 - 1.0)

        e_mj = "ERR"[(typeA.resname,typeB.resname)]        
        epsilon = alpha * np.abs( e_mj - epsilon0 )
        lambda_ = -1 if epsilon0 > e_mj else 1

        sigma = 0.5 * (typeA.sigma + typeB.sigma)
        
        r6 = (sigma/r)**6
        r12 = r6**2
        u_lj = 4 * epsilon * (r12-r6)
        u_hps = lambda_ * np.array(u_lj)
        s = r<=sigma*2**(1/6)
        u_hps[s] = u_lj[s] + (1-lambda_) * epsilon

        u = u_elec + u_hps
        """
        u = u_elec + u_wang
        return u

class MpipiBeads(PolymerBeads):

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

        ## Two consecutive groups 
        if len(ids) == 2:
            b1,b2 = [self.children[i] for i in ids]
            """ units "10 kJ/N_A" kcal_mol """
            bond = HarmonicBond(k = self.spring_constant,
                                r0 = self.rest_length,
                                range_ = (0,100),
                                resolution = 0.01,
                                max_force = 10)

            self.add_bond( i=b1, j=b2, bond = bond, exclude=True )
        elif len(ids) == 3:
            ...
        else:
            pass


class MpipiModel(PolymerModel):
    def __init__(self, polymers,
                 sequences = None,
                 rest_length = 3.8,
                 spring_constant = 2.3900574,
                 debye_length = 7.95,
                 damping_coefficient = 10,
                 DEBUG=False,
                 **kwargs):

        """ 
        [debye_length]: angstroms
        [damping_coefficient]: ns
        """

        logger.info("""You are using an implementation of the Mpipi Polymer model as described for proteins and RNA based on:
        Physics-driven coarse-grained model for biomolecular phase separation with near-quantitative accuracy. Published in final edited form as:
Nat Comput Sci. 2021 Nov; 1(11): 732–743. Published online 2021 Nov 22. doi: 10.1038/s43588-021-00155-3

Please cite all appropriate articles!""")


        if 'timestep' not in kwargs: kwargs['timestep'] = 10e-6
        if 'cutoff' not in kwargs: kwargs['cutoff'] = max(4*debye_length,20)

        if 'decomp_period' not in kwargs:
            kwargs['decomp_period'] = 1000

        self.rest_length = rest_length
        self.spring_constant = spring_constant

        """ Assign sequences """
        if sequences is None:
            raise NotImplementedError("MpipiModel must be provided a sequences argument")

        PolymerModel.__init__(self, polymers, sequences, monomers_per_bead_group=1, **kwargs)


        """ Update type diffusion coefficients """
        self.types = all_types = [t for key,t in _types.items()]
        self.set_damping_coefficient( damping_coefficient )

        """ Set up nonbonded interactions """
        nonbonded = MpipiNonbonded(debye_length)
        for t in all_types:
            self._add_nonbonded_interaction(nonbonded, t)
                
    def _add_nonbonded_interaction(self, interaction, type_):
        i = self.types.index(type_) if type_ in self.types else 0
        for j in range(i,len(self.types)):
            t = self.types[j]
            self.add_nonbonded_interaction( interaction, typeA=type_, typeB=t )

    def _generate_polymer_beads(self, polymer, sequence, polymer_index = None):
        return MpipiBeads(polymer, sequence,
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
    pass
"""
    from matplotlib import pyplot as plt
    nt = len(_types)
    # print("TYPES")
    # for n,t in _types.items():
    #     print("{}\t{}\t{}\t{}\t{}".format(t.name, t.mass, t.charge, t.sigma, t.lambda_))
    type_string = 'WYFMLIVAPGCQNTSEDKHR'
    d = np.zeros([nt,nt])
    for i in range(nt):
        n1 = type_string[i]
        t1 = _types[n1]
        for j in range(nt):
            n2 = type_string[j]
            t2 = _types[n2]
            d[nt-i-1,j] = "ERR"[(t1.name,t2.name)]

    plt.imshow(d.T)
    plt.show()
"""
