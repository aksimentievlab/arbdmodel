# -*- coding: utf-8 -*-
## Test with `python -m arbdmodel.onck_polymer_model`

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
    L = ParticleType("LUE",
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

class OnckNonbonded(NonbondedScheme):
    def __init__(self, debye_length=10, resolution=0.1, rMin=0):
        NonbondedScheme.__init__(self, typesA=None, typesB=None,
                                 resolution=resolution, rMin=rMin)
        self.debye_length = debye_length
        self.maxForce = 50
        # self.maxForce = 100
        # self.maxForce = None

    def potential(self, r, typeA, typeB):
        """ Electrostatics """
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

class OnckBeadsFromPolymer(Group):
    """ units "8038 kJ / (N_A nm**2)" "0.5 * kcal_mol/AA**2" """
    peptide_bond = HarmonicBond(k = 38.422562,
                        r0 = 3.8,
                        rRange = (0,500),
                        resolution = 0.01,
                        maxForce = 10)

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
        
    def _clear_beads(self):
        ...
        
    def _generate_beads(self):
        for i in range(self.polymer.num_monomers):
            c = self.polymer.monomer_index_to_contour(i)
            r = self.polymer.contour_to_position(c)
            s = self.sequence[i]

            bead = PointParticle(_types[s], r,
                                 name = s,
                                 resid = i+1)
            self.add(bead)

        ## Two consecutive monomers 
        for i in range(len(self.children)-1):
            b1,b2 = [self.children[i+j] for j in range(2)]
            bond = OnckBeadsFromPolymer.peptide_bond
            self.add_bond( i=b1, j=b2, bond = bond, exclude=True )

        def bead_to_type(bead):
            if bead.type_.name == 'PRO':
                return 'P'
            elif bead.type_.name == 'GLY':
                return 'G'
            else:
                return 'X'

        ## Three consecutive monomers 
        for i in range(len(self.children)-2):
            b1,b2,b3 = [self.children[i+j] for j in range(3)]
            t1,t2,t3 = ([bead_to_type(b) for b in (b1,b2,b3)])

            filename = 'onck_model_potentials/bend_O{}{}.txt'.format(
                t2,'P' if t3 == 'P' else 'Y' )
            self.add_angle( i=b1, j=b2, k=b3, 
                          angle = get_resource_path(filename) )
            self.add_exclusion( i=b1, j=b3 )

        ## Four consecutive monomers 
        for i in range(len(self.children)-3):
            b1,b2,b3,b4 = [self.children[i+j] for j in range(4)]
            t1,t2,t3,t4 = ([bead_to_type(b) for b in (b1,b2,b3,b4)])

            filename = 'onck_model_potentials/dih_{}{}.txt'.format(t2,t3)
            self.add_dihedral( i=b1, j=b2, k=b3, l=b4,
                               dihedral = get_resource_path(filename) )
            self.add_exclusion( i=b1, j=b4 )

class OnckModel(ArbdModel):
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
            print("""WARNING: you are deviated from the model published by Onck by choosing a debye length != 1 nm.
    Be advised that the non-bonded cutoff is simply set to 5 * debye_length, but this is not necessarily prescribed by the model.""")
        kwargs['timestep'] = 20e-6
        kwargs['cutoff'] = max(5*debye_length,25)

        if 'decompPeriod' not in kwargs:
            kwargs['decompPeriod'] = 1000

        """ Assign sequences """
        if sequences is None:
            raise NotImplementedError("OnckModel must be provided a sequences argument")

        self.polymer_group = AbstractPolymerGroup(polymers)
        self.sequences = sequences
        ArbdModel.__init__(self, [], **kwargs)

        """ Update type diffusion coefficients """
        self.types = all_types = [t for key,t in _types.items()]
        self.set_damping_coefficient( damping_coefficient )

        """ Set up nonbonded interactions """
        nonbonded = OnckNonbonded(debye_length)
        for i in range(len(all_types)):
            t1 = all_types[i]
            for j in range(i,len(all_types)):
                t2 = all_types[j]
                self.useNonbondedScheme( nonbonded, typeA=t1, typeB=t2 )
                
        """ Generate beads """
        self.generate_beads()

    def update_splines(self, coords):
        i = 0
        for p in self.polymer_group.polymers:
            n = p.num_monomers
            p.set_splines(np.linspace(0,1,n), coords[i:i+n])
            i += n

        self.clear_all()
        self.generate_beads()
        ## TODO Apply restraints, etc

    def generate_beads(self):
        self.peptides = [OnckBeadsFromPolymer(p,s)
                         for p,s in zip(self.polymer_group.polymers,self.sequences)]

        for s in self.peptides:
            self.add(s)
            s._generate_beads()

    def set_damping_coefficient(self, damping_coefficient):
        for t in self.types:
            t.damping_coefficient = damping_coefficient
            # t.diffusivity = 831447.2 * temperature / (t.mass * damping_coefficient)

if __name__ == "__main__":

    print("TYPES")
    for n,t in _types.items():
        print("{}\t{}\t{}\t{}\t{}".format(n, t.name, t.mass, t.charge, t.epsilon))
