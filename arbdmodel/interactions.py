from abc import abstractmethod, ABCMeta
import os, sys
import numpy as np
from shutil import copyfile

class NonbondedInteraction(metaclass=ABCMeta):
    """ Abstract class for writing nonbonded interactions """

    def __init__(self, typesA=None, typesB=None, resolution=0.1, rMin=0):
        """If typesA is None, and typesB is None, then """
        self.resolution = resolution
        self.rMin = rMin

    def add_sim_system(self, simSystem):
        self.rMax = simSystem.cutoff
        self.r = np.arange(rMin,rMax,resolution)

    @abstractmethod
    def potential(self, r, typeA, typeB):
        raise NotImplementedError
    
    def __remove_nans(self, u):
        left = np.isnan(u[:-1])
        right = np.where(left)[0]+1
        while np.any(np.isnan(u[right])):
            right[np.isnan(u[right])] += 1
        u[:-1][left] = u[right]

    def write_file(self, filename, typeA, typeB, rMax):
        r = np.arange(self.rMin, rMax+self.resolution, self.resolution)
        with np.errstate(divide='ignore',invalid='ignore'):
            u = self.potential(r, typeA, typeB)
        self.__remove_nans(u)
        np.savetxt(filename, np.array([r,u]).T)


class LennardJones(NonbondedInteraction):
    def potential(self, r, typeA, typeB):
        epsilon = np.sqrt( typeA.epsilon**2 + typeB.epsilon**2 )
        r0 = 0.5 * (typeA.radius + typeB.radius)
        r6 = (r0/r)**6
        r12 = r6**2
        u = 4 * epsilon * (r12-r6)
        # u[0] = u[1]             # Remove NaN
        return u
# LennardJones = LennardJones()

class HalfHarmonic(NonbondedInteraction):
    def potential(self, r, typeA, typeB):
        k = 10                   # kcal/mol AA**2
        r0 = (typeA.radius + typeB.radius)
        u =  0.5 * k * (r-r0)**2
        u[r > r0] = np.zeros( np.shape(u[r > r0]) )
        return u
# HalfHarmonic = HalfHarmonic()

class TabulatedNonbonded(NonbondedInteraction):
    def __init__(self, tableFile, typesA=None, typesB=None, resolution=0.1, rMin=0):
        """If typesA is None, and typesB is None, then """
        self.tableFile = tableFile
        # self.resolution = resolution
        # self.rMin = rMin

        ## TODO: check that tableFile exists and is regular file
        
    def write_file(self, filename, typeA, typeB, rMax):
        if filename != self.tableFile:
            copyfile(self.tableFile, filename)

## Bonded potentials
class BasePotential():
    def __init__(self, r0, rRange=(0,50), resolution=0.1, maxForce=None, max_potential=None, zero="min", prefix="potentials/"):
        self.r0 = r0
        self.rRange = rRange
        self.resolution = resolution
        self.maxForce = maxForce
        self.prefix = prefix
        self.zero = zero
        self.periodic = False
        self.type_ = "None"
        self.max_potential = max_potential
        self.kscale_ = None     # only used for 

    def filename(self):
        raise NotImplementedError("Not implemented")

    def potential(self,dr):
        raise NotImplementedError("Not implemented")

    def __str__(self):
        return self.filename()

    def write_file(self):
        r = np.arange( self.rRange[0], 
                       self.rRange[1]+self.resolution, 
                       self.resolution )
        dr = r-self.r0

        if self.periodic == True:
            rSpan = self.rRange[1]-self.rRange[0]
            assert(rSpan > 0)
            dr = np.mod( dr+0.5*rSpan, rSpan) - 0.5*rSpan 

        u = self.potential(dr)

        maxForce = self.maxForce
        if maxForce is not None:
            assert(maxForce > 0)
            f = np.diff(u)/np.diff(r)
            f[f>maxForce] = maxForce
            f[f<-maxForce] = -maxForce
            u[0] = 0
            u[1:] = np.cumsum(f*np.diff(r))

        if self.max_potential is not None:
            f = np.diff(u)/np.diff(r)
            ids = np.where( 0.5*(u[1:]+u[:-1]) > self.max_potential )[0]

            w = np.sqrt(2*self.max_potential/self.k)
            drAvg = 0.5*(np.abs(dr[ids]) + np.abs(dr[ids+1]))

            f[ids] = f[ids] * np.exp(-(drAvg-w)/(w))
            u[0] = 0
            u[1:] = np.cumsum(f*np.diff(r))

        if self.zero == "min":
            u = u - np.min(u)

        np.savetxt( self.filename(), np.array([r, u]).T, fmt="%f" )

class HarmonicPotential(BasePotential):
    def __init__(self, k, r0, rRange, resolution, maxForce, max_potential, zero="min", correct_geometry=False, prefix='./'):
        self.k = k
        self.kscale_ = None
        BasePotential.__init__(self, r0, rRange, resolution, maxForce, max_potential, zero, prefix)

    def filename(self):
        return "%s%s-%.3f-%.3f.dat" % (self.prefix, self.type_,
                                       self.k*self.kscale_, self.r0)
    def potential(self,dr):
        return 0.5*self.k*dr**2

    def __hash__(self):
        assert(self.type_ != "None")
        return hash((self.type_, self.k, self.r0, self.rRange, self.resolution, self.maxForce, self.max_potential, self.prefix, self.periodic))

    def __eq__(self, other):
        for a in ("type_", "k", "r0", "rRange", "resolution", "maxForce", "max_potential", "prefix", "periodic"):
            if self.__dict__[a] != other.__dict__[a]:
                return False
        return True

# class NonBonded(HarmonicPotential):
#     def _init_hook(self):
#         self.type = "nonbonded"
#         self.kscale_ = 1.0

class HarmonicBond(HarmonicPotential):
    def __init__(self, k, r0, rRange=(0,50), resolution=0.1, maxForce=None, max_potential=None, prefix="potentials/",  zero="min", correct_geometry=False, temperature=295):
        HarmonicPotential.__init__(self, k, r0, rRange, resolution, maxForce, max_potential, zero=zero, correct_geometry=correct_geometry, prefix=prefix)
        self.type_ = "gbond" if correct_geometry else "bond"
        self.kscale_ = 1.0
        self.correct_geometry = correct_geometry
        self.temperature = temperature

    def potential(self,dr):
        u = HarmonicPotential.potential(self,dr)
        if self.correct_geometry:
            with np.errstate(divide='ignore',invalid='ignore'):
                du = 2*0.58622592*np.log(dr+self.r0) * self.temperature/295
            du[np.logical_not(np.isfinite(du))] = 0
            u = u+du
        return u


class HarmonicAngle(HarmonicPotential):
    def __init__(self, k, r0, rRange=(0,181), resolution=0.1, maxForce=None, max_potential=None, zero="min", prefix="potentials/"):
        HarmonicPotential.__init__(self, k, r0, rRange, resolution, maxForce, max_potential, zero=zero, prefix=prefix)
        self.type_ = "angle"
        self.kscale_ = (180.0/np.pi)**2

class HarmonicDihedral(HarmonicPotential):
    def __init__(self, k, r0, rRange=(-180,180), resolution=0.1, maxForce=None, max_potential=None, zero="min", prefix="potentials/"):
        HarmonicPotential.__init__(self, k, r0, rRange, resolution, maxForce, max_potential, zero=zero, prefix=prefix)
        self.periodic = True
        self.type_ = "dihedral"
        self.kscale_ = (180.0/np.pi)**2

class WLCSKBond(BasePotential):
    """ ## https://aip.scitation.org/doi/full/10.1063/1.4968020 """
    def __init__(self, d, lp, kT, rRange=(0,50), resolution=0.02, maxForce=100, max_potential=None, zero="min", prefix="potentials/"):
        BasePotential.__init__(self, 0, rRange, resolution, maxForce, max_potential, zero, prefix)
        self.type_ = "wlcbond"
        self.d = d          # separation
        self.lp = lp            # persistence length
        self.kT = kT

    def filename(self):
        return "%s%s-%.3f-%.3f.dat" % (self.prefix, self.type_,
                                       self.d, self.lp)
    def potential(self, dr):
        nk = self.d / (2*self.lp)
        q2 = (dr / self.d)**2
        a1,a2 = 1, -7.0/(2*nk)
        a3 = 3.0/32 - 3.0/(8*nk) - 6.0/(4*nk**2)
        p0,p1,p2,p3,p4 = 13.0/32, 3.4719,2.5064,-1.2906,0.6482
        a4 = (p0 + p1/(2*nk) + p2*(2*nk)**-2) / (1+p3/(2*nk)+p4*(2*nk)**-2)
        with np.errstate(divide='ignore',invalid='ignore'):
            u = self.kT * nk * ( a1/(1-q2) - a2*np.log(1-q2) + a3*q2 - 0.5*a4*q2*(q2-2) )
        max_force = np.diff(u[q2<1][-2:]) / np.diff(dr).mean()
        max_u = u[q2<1][-1]
        max_dr = dr[q2<1][-2]
        assert( max_force >= 0 )
        u[q2>=1] = (dr[q2>=1]-max_dr)*max_force + max_u
        return u

    def __hash__(self):
        assert(self.type_ != "None")
        return hash((self.type_, self.d, self.lp, self.kT, self.rRange, self.resolution, self.maxForce, self.max_potential, self.prefix, self.periodic))

    def __eq__(self, other):
        for a in ("type_", "d", "lp", "kT" "rRange", "resolution", "maxForce", "max_potential", "prefix", "periodic"):
            if self.__dict__[a] != other.__dict__[a]:
                return False
        return True

class WLCSKAngle(BasePotential):
    ## https://aip.scitation.org/doi/full/10.1063/1.4968020
    def __init__(self, d, lp, kT, rRange=(0,181), resolution=0.5, maxForce=None, max_potential=None, zero="min", prefix="potentials/"):
        BasePotential.__init__(self, 180, rRange, resolution, maxForce, max_potential, zero, prefix)
        self.type_ = "wlcangle"
        self.d = d          # separation
        self.lp = lp            # persistence length
        self.kT = kT

    def filename(self):
        return "%s%s-%.3f-%.3f.dat" % (self.prefix, self.type_,
                                       self.d, self.lp)
    def potential(self,dr):
        nk = self.d / (2*self.lp)
        p1,p2,p3,p4 = -1.237, 0.8105, -1.0243, 0.4595
        C = (1 + p1*(2*nk) + p2*(2*nk)**2) / (2*nk+p3*(2*nk)**2+p4*(2*nk)**3)
        u = self.kT * C * (1-np.cos(dr * np.pi / 180))
        return u

    def __hash__(self):
        assert(self.type_ != "None")
        return hash((self.type_, self.d, self.lp, self.kT, self.rRange, self.resolution, self.maxForce, self.max_potential, self.prefix, self.periodic))

    def __eq__(self, other):
        for a in ("type_", "d", "lp", "kT" "rRange", "resolution", "maxForce", "max_potential", "prefix", "periodic"):
            if self.__dict__[a] != other.__dict__[a]:
                return False
        return True
