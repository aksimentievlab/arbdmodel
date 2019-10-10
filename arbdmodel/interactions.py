from shutil import copyfile
import os, sys
import numpy as np

class NonbondedScheme():
    """ Abstract class for writing nonbonded interactions """

    def __init__(self, typesA=None, typesB=None, resolution=0.1, rMin=0):
        """If typesA is None, and typesB is None, then """
        self.resolution = resolution
        self.rMin = rMin

    def add_sim_system(self, simSystem):
        self.rMax = simSystem.cutoff
        self.r = np.arange(rMin,rMax,resolution)

    def potential(self, r, typeA, typeB):
        raise NotImplementedError
    
    def write_file(self, filename, typeA, typeB, rMax):
        r = np.arange(self.rMin, rMax+self.resolution, self.resolution)
        u = self.potential(r, typeA, typeB)
        np.savetxt(filename, np.array([r,u]).T)


class LennardJones(NonbondedScheme):
    def potential(self, r, typeA, typeB):
        epsilon = np.sqrt( typeA.epsilon**2 + typeB.epsilon**2 )
        r0 = 0.5 * (typeA.radius + typeB.radius)
        r6 = (r0/r)**6
        r12 = r6**2
        u = epsilon * (r12-2*r6)
        u[0] = u[1]             # Remove NaN
        return u
# LennardJones = LennardJones()

class HalfHarmonic(NonbondedScheme):
    def potential(self, r, typeA, typeB):
        k = 10                   # kcal/mol AA**2
        r0 = (typeA.radius + typeB.radius)
        u =  0.5 * k * (r-r0)**2
        u[r > r0] = np.zeros( np.shape(u[r > r0]) )
        return u
# HalfHarmonic = HalfHarmonic()

class TabulatedPotential(NonbondedScheme):
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
class HarmonicPotential():
    def __init__(self, k, r0, rRange=(0,50), resolution=0.1, maxForce=None, max_potential=None, prefix="potentials/"):
        self.k = k
        self.r0 = r0
        self.rRange = rRange
        self.resolution = 0.1
        self.maxForce = maxForce
        self.prefix = prefix
        self.periodic = False
        self.type_ = "None"
        self.max_potential = max_potential
        self.kscale_ = None     # only used for 

    def filename(self):
        # raise NotImplementedError("Not implemented")
        return "%s%s-%.3f-%.3f.dat" % (self.prefix, self.type_,
                                       self.k*self.kscale_, self.r0)

    def __str__(self):
        return self.filename()

    def potential(self, dr):
        return 0.5*self.k*dr**2

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

        u = u - np.min(u)

        np.savetxt( self.filename(), np.array([r, u]).T, fmt="%f" )

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
    def __init__(self, k, r0, rRange=(0,50), resolution=0.1, maxForce=None, max_potential=None, prefix="potentials/"):
        HarmonicPotential.__init__(self, k, r0, rRange, resolution, maxForce, max_potential, prefix)
        self.type_ = "bond"
        self.kscale_ = 1.0

class HarmonicAngle(HarmonicPotential):
    def __init__(self, k, r0, rRange=(0,181), resolution=0.1, maxForce=None, max_potential=None, prefix="potentials/"):
        HarmonicPotential.__init__(self, k, r0, rRange, resolution, maxForce, max_potential, prefix)
        self.type_ = "angle"
        self.kscale_ = (180.0/np.pi)**2

class HarmonicDihedral(HarmonicPotential):
    def __init__(self, k, r0, rRange=(-180,180), resolution=0.1, maxForce=None, max_potential=None, prefix="potentials/"):
        HarmonicPotential.__init__(self, k, r0, rRange, resolution, maxForce, max_potential, prefix)
        self.periodic = True
        self.type_ = "dihedral"
        self.kscale_ = (180.0/np.pi)**2

