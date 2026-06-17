## Test with `python -m arbdmodel.calvados3_polymer_model`

import numpy as np
## Local imports
from . import logger, ParticleType, PointParticle
from .polymer import PolymerBeads, PolymerModel
from .interactions import AbstractPotential, HarmonicBond

"""Define particle types"""
_types = dict(
    A = ParticleType("ALA",
                     mass = 71.07,
                     charge = 0,
                     sigma = 5.04,
                     lambda_=0.3377244362031627
                 ),
    R = ParticleType("ARG",
                     mass = 156.19,
                     charge = 1,
                     sigma = 6.56,
                     lambda_= 0.7407902764839954	
                 ),
    N = ParticleType("ASN",
                     mass = 114.1,
                     charge = 0,
                     sigma = 5.68,
                     lambda_= 0.3706962163690402
                 ),
    D = ParticleType("ASP",
                     mass = 115.09,
                     charge = -1,
                     sigma = 5.58,
                     lambda_=0.092587557536158,
                 ),
    C = ParticleType("CYS",
                     mass = 103.14,
                     charge = 0,
                     sigma = 5.48,
                     lambda_=0.5922529084601322
                 ),
    Q = ParticleType("GLN",
                     mass = 128.13,
                     charge = 0,
                     sigma = 6.02,
                     lambda_=0.3143449791669133,
                 ),
    E = ParticleType("GLU",
                     mass = 129.11,
                     charge = -1,
                     sigma = 5.92,
                     lambda_=0.3706962163690402,
                 ),
    G = ParticleType("GLY",
                     mass = 57.05,
                     charge = 0,
                     sigma = 4.5,
                     lambda_=0.7538308115197386
                 ),
    H = ParticleType("HIS",
                     mass = 137.14,
                     charge = 0,
                     sigma = 6.08,
                     lambda_=0.4087176216525476
                 ),
    I = ParticleType("ILE",
                     mass = 113.16,
                     charge = 0,
                     sigma = 6.18,
                     lambda_=0.5130398874425708
                 ),
    L = ParticleType("LEU",
                     mass = 113.16,
                     charge = 0,
                     sigma = 6.18,
                     lambda_=0.5548615312993875
                 ),
    K = ParticleType("LYS",
                     mass = 128.17,
                     charge = 1,
                     sigma = 6.36,
                     lambda_=0.1380602542039267
                 ),
    M = ParticleType("MET",
                     mass = 131.2,
                     charge = 0,
                     sigma = 6.18,
                     lambda_=0.5170874160398543
                 ),
    F = ParticleType("PHE",
                     mass = 147.18,
                     charge = 0,
                     sigma = 6.36,
                     lambda_=0.8906449355499866
                 ),
    P = ParticleType("PRO",
                     mass = 97.12,
                     charge = 0,
                     sigma = 5.56,
                     lambda_=0.3469777523519372
                 ),
    S = ParticleType("SER",
                     mass = 87.08,
                     charge = 0,
                     sigma = 5.18,
                     lambda_=0.4473142572693176
                 ),
    T = ParticleType("THR",
                     mass = 101.11,
                     charge = 0,
                     sigma = 5.62,
                     lambda_=0.2672387936544146,
                 ),
    W = ParticleType("TRP",
                     mass = 186.22,
                     charge = 0,
                     sigma = 6.78,
                     lambda_=1.033450123574512,
                 ),
    Y = ParticleType("TYR",
                     mass = 163.18,
                     charge = 0,
                     sigma = 6.46,
                     lambda_=0.950628687301107
                 ),
    V = ParticleType("VAL",
                     mass = 99.13,
                     charge = 0,
                     sigma = 5.86,
                     lambda_=0.2936174211771383
                 )
)

for k,t in list(_types.items()):
    t.resname = t.name
    t.is_idp = False
    
    ## Add types for IDPs

    
class CalvadosNonbonded(AbstractPotential):
    def __init__(self, debye_length=10, resolution=0.1, range_=(0,None),temperature=293.15):
        AbstractPotential.__init__(self, resolution=resolution, range_=range_)
        self.debye_length = debye_length
        self.max_force = 50
        self.temperature = temperature
    def potential(self, r, types):
        """ Electrostatics """
        typeA, typeB = types
        ld = self.debye_length 
        q1 = typeA.charge
        q2 = typeB.charge
        T = self.temperature
        D = 5321/T + 233.76-0.9297*T+1.417*1e-3*T**2 - 8.292*1e-7*T**3                 # dielectric of water
        ## units "e**2 / (4 * pi * epsilon0 AA)" kcal_mol
        A =  332.06371
        rc = 40 ##4nm
        u_elec = np.zeros_like(r,dtype = float)
        mask = r<rc
        u_elec_r = (A*q1*q2/D)*np.exp(-r/ld) / r 
        u_elec_rc= (A*q1*q2/D)*np.exp(-rc/ld) / rc 
        u_elec[mask] = u_elec_r[mask] - u_elec_rc
            
        """Ashbaugh-Hatch potential"""
        sigma = 0.5 * (typeA.sigma + typeB.sigma)
        lambda_ = 0.5 * (typeA.lambda_ + typeB.lambda_)
        r6 = (sigma/r)**6
        r12 = r6**2
        epsilon = 0.2
        u_lj = 4 * epsilon * (r12-r6)
        rc = 20
        u_lj_rc = 4*epsilon*((sigma/rc)**12 - (sigma/rc)**6)
        u_AH = np.zeros_like(r,dtype=float)
        mask1 = r<=2**(1/6.0)*sigma
        mask2 = (r>2**(1/6.0)*sigma) & (r<=rc)
        u_AH[mask1] = u_lj[mask1] - lambda_*u_lj_rc + epsilon*(1-lambda_)
        u_AH[mask2] = lambda_*(u_lj[mask2] - u_lj_rc)
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
        u = u_elec + u_AH
        return u

class CalvadosBeads(PolymerBeads):

    def __init__(self, polymer, sequence=None,
                 spring_constant = 19.19,
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


class CalvadosModel(PolymerModel):
    def __init__(self, polymers, 
                 sequences = None,
                 rest_length = 3.8,
                 spring_constant = 19.19,
                 debye_length = 7.95,
                 damping_coefficient = 10,
                 DEBUG=False,
                 **kwargs):

        """ 
        [debye_length]: angstroms
        [damping_coefficient]: 1/ns (zeta/m, written to .bd as transDamping)
        """

        logger.info("""You are using an implementation of the Calvados3 Polymer model as described for proteins:
        A coarse-grained model for disordered and multi-domain proteins. Published in final edited form as:
Protein Science 33, no. 11 (2024): e5172. Published online 2024 Oct 16. doi:https://doi.org/10.1002/pro.5172
Please cite all appropriate articles!""")


        if 'timestep' not in kwargs: kwargs['timestep'] = 10e-6
        if 'cutoff' not in kwargs: kwargs['cutoff'] = max(4*debye_length,20)
        if 'temperature' not in kwargs:  kwargs['temperature'] = 293.15
        if 'decomp_period' not in kwargs:
            kwargs['decomp_period'] = 1000
        self.temperature = kwargs['temperature']
        self.rest_length = rest_length
        self.spring_constant = spring_constant 
        """ Assign sequences """
        if sequences is None:
            raise NotImplementedError("Calvados3Model must be provided a sequences argument")

        PolymerModel.__init__(self, polymers, sequences, monomers_per_bead_group=1, **kwargs)


        """ Update type diffusion coefficients """
        self.types = all_types = [t for key,t in _types.items()]
        self.set_damping_coefficient( damping_coefficient )

        """ Set up nonbonded interactions """
        nonbonded = CalvadosNonbonded(debye_length,temperature=self.temperature)
        for t in all_types:
            self._add_nonbonded_interaction(nonbonded, t)
                
    def _add_nonbonded_interaction(self, interaction, type_):
        i = self.types.index(type_) if type_ in self.types else 0
        for j in range(i,len(self.types)):
            t = self.types[j]
            self.add_nonbonded_interaction( interaction, typeA=type_, typeB=t )

    def _generate_polymer_beads(self, polymer, sequence, polymer_index = None):
        return CalvadosBeads(polymer, sequence,
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
