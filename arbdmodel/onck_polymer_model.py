# -*- coding: utf-8 -*-
## Test with `python -m arbdmodel.onck_polymer_model`

import numpy as np
import sys


## Local imports
from . import logger, ParticleType, PointParticle, get_resource_path
from .polymer import PolymerBeads, PolymerModel
from .interactions import AbstractPotential, HarmonicBond

"""Define particle types"""
_types = dict(
    A = ParticleType("ALA",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.7,
                     lambda_ = 0.72973,
                 ),
    R = ParticleType("ARG",
                     mass = 120,
                     charge = 1,
                     epsilon = 0.0,
                     lambda_ = 0.0,
                 ),
    N = ParticleType("ASN",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.33,
                     lambda_ = 0.432432,
                 ),
    D = ParticleType("ASP",
                     mass = 120,
                     charge = -1,
                     epsilon = 0.0005,
                 ),
    C = ParticleType("CYS",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.68,
                 ),
    Q = ParticleType("GLN",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.64,
                 ),
    E = ParticleType("GLU",
                     mass = 120,
                     charge = -1,
                     epsilon = 0.0005,
                 ),
    G = ParticleType("GLY",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.41,
                 ),
    H = ParticleType("HIS",
                     mass = 120,
                     charge = 0.0,
                     epsilon = 0.53,
                 ),
    I = ParticleType("ILE",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.98,
                 ),
    L = ParticleType("LEU",
                     mass = 120,
                     charge = 0,
                     epsilon = 1.0,
                 ),
    K = ParticleType("LYS",
                     mass = 120,
                     charge = 1,
                     epsilon = 0.0005,
                 ),
    M = ParticleType("MET",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.78,
                 ),
    F = ParticleType("PHE",
                     mass = 120,
                     charge = 0,
                     epsilon = 1.0,
                 ),
    P = ParticleType("PRO",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.65,
                 ),
    S = ParticleType("SER",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.45,
                 ),
    T = ParticleType("THR",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.51,
                 ),
    W = ParticleType("TRP",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.96,
                 ),
    Y = ParticleType("TYR",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.82,
                 ),
    V = ParticleType("VAL",
                     mass = 120,
                     charge = 0,
                     epsilon = 0.94,
                 )
)
for k,t in _types.items():
    t.resname = t.name

class OnckNonbonded(AbstractPotential):
    def __init__(self, debye_length=10, resolution=0.1, range_=(0,None)):
        AbstractPotential.__init__(self, resolution=resolution, range_=range_)
        self.debye_length = debye_length
        self.max_force = 50

    def potential(self, r, types):
        """ Electrostatics """
        typeA, typeB = types
        ld = self.debye_length
        q1 = typeA.charge
        q2 = typeB.charge
        D = 80                  # dielectric of water
        _z = 2.5
        D = 80 * (1- (r/_z)**2 * np.exp(r/_z)/(np.exp(r/_z)-1)**2)
        ## units "e**2 / (4 * pi * epsilon0 AA)" kcal_mol
        A =  332.06371
        u_elec = (A*q1*q2/D)*np.exp(-r/ld) / r

        """ LJ-type term """
        alpha = 0.27
        epsilon_hp = 3.1070746 # units "13 kJ/N_A" kcal_mol
        epsilon_rep = 2.3900574 # units "10 kJ/N_A" kcal_mol

        sigma = 6.0
        epsilon = epsilon_hp*np.sqrt( (typeA.epsilon*typeB.epsilon)**alpha )

        r6 = (sigma/r)**6
        r8 = (sigma/r)**8
        u_lj = (epsilon_rep-epsilon) * r8
        s = r<=sigma
        u_lj[s] = epsilon_rep*r8[s] - epsilon*(4*r6[s]-1)/3
        u_lj[r>25] = 0
        
        u = u_elec + u_lj
        return u

class OnckBeads(PolymerBeads):
    
 
    def __init__(self, polymer, sequence=None,
                 spring_constant = 38.422562, # units "8038 kJ / (N_A nm**2)" "0.5 * kcal_mol/AA**2"
                 rest_length=3.8, **kwargs):

        self.peptide_bond = HarmonicBond(k = spring_constant,
                                         r0 = rest_length,
                                         range_ = (0,100),
                                         resolution = 0.01,
                                         max_force = 10)

        if sequence is None:
            raise NotImplementedError
            # ... set random sequence

        self.polymer = polymer
        self.sequence = sequence

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

        def bead_to_type(bead):
            if bead.type_.name == 'PRO':
                return 'P'
            elif bead.type_.name == 'GLY':
                return 'G'
            else:
                return 'X'

        ## Two consecutive groups 
        if len(ids) == 2:
            b1,b2 = [self.children[i] for i in ids]
            """ units "10 kJ/N_A" kcal_mol """
            bond = self.peptide_bond
            self.add_bond( i=b1, j=b2, bond = bond, exclude=True )
        elif len(ids) == 3:
            b1,b2,b3 = [self.children[i] for i in ids]
            t1,t2,t3 = [bead_to_type(b) for b in (b1,b2,b3)]

            filename = 'onck_model_potentials/bend_O{}{}.txt'.format(
                t2, 'P' if t3 == 'P' else 'Y' )
            self.add_angle( i=b1, j=b2, k=b3, 
                          angle = get_resource_path(filename) )
            self.add_exclusion( i=b1, j=b3 )
        elif len(ids) == 4:
            ## Four consecutive monomers
            b1,b2,b3,b4 = [self.children[i] for i in ids]
            t1,t2,t3,t4 = [bead_to_type(b) for b in (b1,b2,b3,b4)]

            filename = 'onck_model_potentials/dih_{}{}.txt'.format(t2,t3)
            self.add_dihedral( i=b1, j=b2, k=b3, l=b4,
                               dihedral = get_resource_path(filename) )
            self.add_exclusion( i=b1, j=b4 )
        else:
            raise Exception('Programming error!')

class OnckModel(PolymerModel):
    def __init__(self, polymers,
                 sequences = None,
                 debye_length = 10,
                 damping_coefficient = 50e3,
                 DEBUG=False,
                 **kwargs):

        """ 
        [debye_length]: angstroms
        [damping_coefficient]: ns
        """
        if debye_length != 10:
            logger.warning("""Deviating from the model published by Onck by choosing a debye length != 1 nm.
    Be advised that the non-bonded cutoff is simply set to 5 * debye_length, but this is not necessarily prescribed by the model.""")

        if 'timestep' not in kwargs: kwargs['timestep'] = 20e-6
        if 'cutoff' not in kwargs: kwargs['cutoff'] = max(5*debye_length,25)

        if 'decomp_period' not in kwargs:
            kwargs['decomp_period'] = 1000

        """ Assign sequences """
        if sequences is None:
            raise NotImplementedError("OnckModel must be provided a sequences argument")

        PolymerModel.__init__(self, polymers, sequences, monomers_per_bead_group=1, **kwargs)

        """ Update type diffusion coefficients """
        self.types = all_types = [t for key,t in _types.items()]
        self.set_damping_coefficient( damping_coefficient )

        """ Set up nonbonded interactions """
        nonbonded = OnckNonbonded(debye_length)
        for t in all_types:
            self._add_nonbonded_interaction(nonbonded, t)
                
    def _add_nonbonded_interaction(self, interaction, type_):
        i = self.types.index(type_) if type_ in self.types else 0
        for t in self.types[i:]:
            self.add_nonbonded_interaction( interaction, typeA=type_, typeB=t )

    def _generate_polymer_beads(self, polymer, sequence):
        return OnckBeads(polymer, sequence,
                       monomers_per_bead_group = self.monomers_per_bead_group,
                       )

    def set_damping_coefficient(self, damping_coefficient):
        for t in self.types:
            t.damping_coefficient = damping_coefficient
            # t.diffusivity = 831447.2 * temperature / (t.mass * damping_coefficient)

if __name__ == "__main__":

    print("TYPES")
    for n,t in _types.items():
        print("{}\t{}\t{}\t{}\t{}".format(n, t.name, t.mass, t.charge, t.epsilon))
