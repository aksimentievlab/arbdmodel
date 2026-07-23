# -*- coding: utf-8 -*-
## Test with `python -m arbdmodel.kh_polymer_model`

import numpy as np

## Local imports
from .logger import logger, get_resource_path
from . import ParticleType, PointParticle
from .polymer import PolymerBeads, PolymerModel
from .interactions import AbstractPotential, HarmonicBond

"""Define particle types"""
_types = dict(
    A = ParticleType("ALA",
                     mass = 71.08,
                     charge = 0,
                     sigma = 5.04,
                 ),
    R = ParticleType("ARG",
                     mass = 156.2,
                     charge = 1,
                     sigma = 6.56,
                 ),
    N = ParticleType("ASN",
                     mass = 114.1,
                     charge = 0,
                     sigma = 5.68,
                 ),
    D = ParticleType("ASP",
                     mass = 115.1,
                     charge = -1,
                     sigma = 5.58,
                 ),
    C = ParticleType("CYS",
                     mass = 103.1,
                     charge = 0,
                     sigma = 5.48,
                 ),
    Q = ParticleType("GLN",
                     mass = 128.1,
                     charge = 0,
                     sigma = 6.02,
                 ),
    E = ParticleType("GLU",
                     mass = 129.1,
                     charge = -1,
                     sigma = 5.92,
                 ),
    G = ParticleType("GLY",
                     mass = 57.05,
                     charge = 0,
                     sigma = 4.5,
                 ),
    H = ParticleType("HIS",
                     mass = 137.1,
                     charge = 0.5,
                     sigma = 6.08,
                 ),
    I = ParticleType("ILE",
                     mass = 113.2,
                     charge = 0,
                     sigma = 6.18,
                 ),
    L = ParticleType("LEU",
                     mass = 113.2,
                     charge = 0,
                     sigma = 6.18,
                 ),
    K = ParticleType("LYS",
                     mass = 128.2,
                     charge = 1,
                     sigma = 6.36,
                 ),
    M = ParticleType("MET",
                     mass = 131.2,
                     charge = 0,
                     sigma = 6.18,
                 ),
    F = ParticleType("PHE",
                     mass = 147.2,
                     charge = 0,
                     sigma = 6.36,
                 ),
    P = ParticleType("PRO",
                     mass = 97.12,
                     charge = 0,
                     sigma = 5.56,
                 ),
    S = ParticleType("SER",
                     mass = 87.08,
                     charge = 0,
                     sigma = 5.18,
                 ),
    T = ParticleType("THR",
                     mass = 101.1,
                     charge = 0,
                     sigma = 5.62,
                 ),
    W = ParticleType("TRP",
                     mass = 186.2,
                     charge = 0,
                     sigma = 6.78,
                 ),
    Y = ParticleType("TYR",
                     mass = 163.2,
                     charge = 0,
                     sigma = 6.46,
                 ),
    V = ParticleType("VAL",
                     mass = 99.07,
                     charge = 0,
                     sigma = 5.86,
                 )
)

for k,t in list(_types.items()):
    t.resname = t.name
    t.is_idp = False
    
    ## Add types for IDPs
    _types[k+'IDP'] = ParticleType(t.name+'IDP', mass=t.mass, charge=t.charge, sigma=t.sigma, is_idp=True, resname=t.resname)


data= np.loadtxt(get_resource_path("kh_pairwise_epsilon.csv"), dtype=str)
epsilon_mj = dict()
for n1,n2,v in data:
    v = float(v)
    epsilon_mj[(n1,n2)] = v
    epsilon_mj[(n2,n1)] = v
    epsilon_mj[(n1+'IDP',n2)]=v
    epsilon_mj[(n2,n1+'IDP')]=v
    epsilon_mj[(n2+'IDP',n1)]=v
    epsilon_mj[(n1,n2+'IDP')]=v
    epsilon_mj[(n2+'IDP',n1+'IDP')]=v
    epsilon_mj[(n1+'IDP',n2+'IDP')]=v


    
class KhNonbonded(AbstractPotential):
    """
    A class implementing the Kim-Hummer nonbonded potential energy model for protein interactions.
    This model combines electrostatic interactions with a modified Lennard-Jones potential
    that accounts for hydrophobic effects using the KH (Kim-Hummer) scaling approach.
    The potential is particularly useful for modeling interactions between intrinsically
    disordered proteins (IDPs) and folded proteins.

    Parameters
    ----------
    debye_length : float, optional
        The Debye screening length in Angstroms, default is 10Å
    resolution : float, optional
        The spatial resolution for potential calculations, default is 0.1Å
    range_ : tuple, optional
        The range of distances (min, max) over which to compute the potential,
        default is (0, None) where None means no upper limit

    Attributes
    ----------
    debye_length : float
        The Debye screening length in Angstroms
    max_force : float
        Maximum allowed force from this potential, set to 50

    Methods
    -------
    potential(r, types)
        Calculates the nonbonded potential energy between two atom types at distance r.
        Combines electrostatic interactions with a modified Lennard-Jones potential.
        
    References
    ----------
    Kim, Y. C., & Hummer, G. (2008). Coarse-grained models for simulations of 
    multiprotein complexes: application to ubiquitin binding. Journal of molecular 
    biology, 375(5), 1416-1433.
    """
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
        ## units "e**2 / (4 * pi * epsilon0 AA)" kcal_mol
        A =  332.06371
        u_elec = (A*q1*q2/D)*np.exp(-r/ld) / r 
        
        """ KH scale model """
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

        e_mj = epsilon_mj[(typeA.resname,typeB.resname)]        
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
        return u

class KhBeads(PolymerBeads):

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

        try:
            polymer.idp_array
            self.idp_array = polymer.idp_array
        except:
            logger.warning("KhBeads processing a polymer without 'idp_array' attribute set (boolean numpy array with one True/False value per amino acid with True corresponding to IDP... Assuming all amino acids are IDP.")
            self.idp_array = np.ones(polymer.num_monomers, dtype=bool)

        if len(self.idp_array) != polymer.num_monomers:
            raise ValueError(f'polymer {polymer} idp_array has incorrect size != {polymer.num_monomers}')

       

    def _generate_ith_bead_group(self, i, r, o):
        s = self.sequence[i]
        if self.idp_array[i]:
            s = s + 'IDP'
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


class KhModel(PolymerModel):
    def __init__(self, polymers,
                 sequences = None,
                 rest_length = 3.8,
                 spring_constant = 2.3900574,
                 debye_length = 10,
                 damping_coefficient = 10,
                 idp_array = None,
                 DEBUG=False,
                 **kwargs):

        """ 
        [debye_length]: angstroms
        [damping_coefficient]: 1/ns (zeta/m, written to .bd as transDamping)
        """

        logger.info("""You are using an implementation of the Kim-Hummer model as described for proteins with IDPs by the Mittal lab:

Dignon GL, Zheng W, Kim YC, Best RB, Mittal J (2018) Sequence determinants of protein phase behavior from a coarse-grained model. PLOS Computational Biology 14(1) e1005941. https://doi.org/10.1371/journal.pcbi.1005941

based upon:

Young C. Kim, Gerhard Hummer (2008) Coarse-grained Models for Simulations of Multiprotein Complexes: Application to Ubiquitin Binding. Journal of Molecular Biology 375(5) 1416-1433. https://doi.org/10.1016/j.jmb.2007.11.063.

Please cite all appropriate articles!""")


        if 'timestep' not in kwargs: kwargs['timestep'] = 10e-6
        if 'cutoff' not in kwargs: kwargs['cutoff'] = max(4*debye_length,20)

        if 'decomp_period' not in kwargs:
            kwargs['decomp_period'] = 1000

        self.rest_length = rest_length
        self.spring_constant = spring_constant

        """ Assign sequences """
        if sequences is None:
            raise NotImplementedError("KhModel must be provided a sequences argument")

        PolymerModel.__init__(self, polymers, sequences, monomers_per_bead_group=1, **kwargs)


        """ Update type diffusion coefficients """
        self.types = all_types = [t for key,t in _types.items()]
        self.set_damping_coefficient( damping_coefficient )

        """ Set up nonbonded interactions """
        nonbonded = KhNonbonded(debye_length)
        for t in all_types:
            self._add_nonbonded_interaction(nonbonded, t)
                
    def _add_nonbonded_interaction(self, interaction, type_):
        i = self.types.index(type_) if type_ in self.types else 0
        for j in range(i,len(self.types)):
            t = self.types[j]
            self.add_nonbonded_interaction( interaction, typeA=type_, typeB=t )

    def _generate_polymer_beads(self, polymer, sequence, polymer_index = None):
        return KhBeads(polymer, sequence,
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
            d[nt-i-1,j] = epsilon_mj[(t1.name,t2.name)]

    plt.imshow(d.T)
    plt.show()
