from abc import abstractmethod, ABCMeta
import os, sys
import numpy as np
from shutil import copyfile

""" Module providing classes used to describe potentials in ARBD """


## Abstract classes that others inherit from
class AbstractPotential(metaclass=ABCMeta):
    """ Abstract class for writing potentials """

    def __init__(self, range_=(0,None), resolution=0.1, 
                 max_force = None, max_potential = None, zero='last'):
        self.range_ = range_
        self.resolution = resolution
        self.max_force = max_force
        self.max_potential = max_potential
        self.zero = zero

    @property
    def range_(self):
        return self.__range_
    @range_.setter
    def range_(self,value):
        assert(len(value) == 2)
        self.__range_ = tuple(value)

    @property
    def periodic(self):
        return False

    @abstractmethod
    def potential(self, r, types=None):
        raise NotImplementedError
    
    def __remove_nans(self, u):
        left = np.isnan(u[:-1])
        right = np.where(left)[0]+1
        while np.any(np.isnan(u[right])):
            right[np.isnan(u[right])] += 1
        u[:-1][left] = u[right]

    def filename(self, types=None):
        raise NotImplementedError('Inherited potential objects should overload this function')

    def _cap_potential(self, r, u):
        self.__remove_nans(u)

        if self.zero == 'min':
            u = u - np.min(u)
        elif self.zero == 'first':
            u = u - u[0]
        elif self.zero == 'last':
            u = u - u[-1]
        else:
            raise ValueError('Unrecognized option for "zero" argument')
        
        max_force = self.max_force
        if max_force is not None:
            assert(max_force > 0)
            f = np.diff(u)/np.diff(r)
            f[f>max_force] = max_force
            f[f<-max_force] = -max_force
            u[0] = 0
            u[1:] = np.cumsum(f*np.diff(r))

        if self.max_potential is not None:
            f = np.diff(u)/np.diff(r)
            ids = np.where( 0.5*(u[1:]+u[:-1]) > self.max_potential )[0]

            # w = np.sqrt(2*self.max_potential/self.k)
            # drAvg = 0.5*(np.abs(dr[ids]) + np.abs(dr[ids+1]))
            # f[ids] = f[ids] * np.exp(-(drAvg-w)/(w))
            f[ids] = self.max_potential
            
            u[0] = 0
            u[1:] = np.cumsum(f*np.diff(r))

        if self.zero == 'min':
            u = u - np.min(u)
        elif self.zero == 'first':
            u = u - u[0]
        elif self.zero == 'last':
            u = u - u[-1]
        else:
            raise ValueError('Unrecognized option for "zero" argument')
        return u

    def write_file(self, filename=None, types=None):
        if filename is None:
            filename = self.filename(types)
        rmin,rmax = self.range_
        r = np.arange(rmin, rmax+self.resolution, self.resolution)
        with np.errstate(divide='ignore',invalid='ignore'):
            u = self.potential(r, types)

        u = self._cap_potential(r,u)

        np.savetxt(filename, np.array([r,u]).T)

    def __hash__(self):
        return hash((self.range_, self.resolution, self.max_force, self.max_potential, self.zero))

    def __eq__(self, other):
        for a in ("range_", "resolution", "max_force", "max_potential", "zero"):
            if getattr(self,a) != getattr(other,a):
                return False
        return type(self).__name__ == type(other).__name__
    
## Concrete nonbonded pontentials
class LennardJones(AbstractPotential):
    def potential(self, r, types):
        typeA, typeB = types
        epsilon = np.sqrt( typeA.epsilon**2 + typeB.epsilon**2 )
        r0 = 0.5 * (typeA.radius + typeB.radius)
        r6 = (r0/r)**6
        r12 = r6**2
        u = 4 * epsilon * (r12-r6)
        # u[0] = u[1]             # Remove NaN
        return u

class HalfHarmonic(AbstractPotential):
    def potential(self, r, types):
        typeA, typeB = types
        k = 10                   # kcal/mol AA**2
        r0 = (typeA.radius + typeB.radius)
        u =  0.5 * k * (r-r0)**2
        u[r > r0] = np.zeros( np.shape(u[r > r0]) )
        return u

class TabulatedNonbonded(AbstractPotential):
    def __init__(self, tableFile, *args, **kwargs):
        self.tableFile = tableFile
        AbstractPotential.__init__(self,*args,**kwargs)

        ## TODO: check that tableFile exists and is regular file

    def potential(self, r, types):
        raise NotImplementedError('This should probably not be implemented')
        
    def write_file(self, filename, types):
        if filename != self.tableFile:
            copyfile(self.tableFile, filename)

## Bonded potentials            
class HarmonicBondedPotential(AbstractPotential):
    def __init__(self, k, r0, filename_prefix='./potentials/', *args, **kwargs):
        self.k = k
        self.r0 = r0
        self.kscale_ = None
        self.filename_prefix = filename_prefix
        if 'zero' not in kwargs: kwargs['zero'] = 'min'
        AbstractPotential.__init__(self, *args, **kwargs)

    @property
    def kscale(self):
        return 1.0
 
    @property
    @abstractmethod
    def type_(self):
        return "none"
   
    def filename(self, types=None):
        assert(types is None)
        return f"{self.filename_prefix}{self.type_}-{self.k*self.kscale:.3f}-{self.r0:.3f}.dat"
    
    def potential(self, r, types=None):
        dr = r-self.r0
        if self.periodic == True:
            r_span = self.range_[1]-self.range_[0]
            assert(r_span > 0)
            dr = np.mod( dr+0.5*r_span, r_span) - 0.5*r_span 
        return 0.5*self.k*dr**2

    def __hash__(self):
        assert(self.type_ != "None")
        return hash((self.type_, self.k, self.r0, self.filename_prefix, self.periodic, AbstractPotential.__hash__(self))) # 

    def __eq__(self, other):
        for a in ("type_", "k", "r0", "filename_prefix", "periodic"):
            if getattr(self,a) != getattr(other,a):
                return False
        return AbstractPotential.__eq__(self,other)
    
class HarmonicBond(HarmonicBondedPotential):
    def __init__(self, k, r0, correct_geometry=False, temperature=295, *args, **kwargs):
        if 'range_' not in kwargs: kwargs['range_'] = (0,50)
        HarmonicBondedPotential.__init__(self, k, r0, *args, **kwargs)
        self.correct_geometry = correct_geometry
        self.temperature = temperature

    @property
    def kscale(self):
        return 1.0
 
    @property
    def type_(self):
        return 'gbond' if self.correct_geometry else 'bond'

    def potential(self, r, types=None):
        u = HarmonicBondedPotential.potential(self, r, types)
        if self.correct_geometry:
            with np.errstate(divide='ignore',invalid='ignore'):
                du = 2*0.58622592*np.log(r) * self.temperature/295
            du[np.logical_not(np.isfinite(du))] = 0
            u = u+du
        return u

class HarmonicAngle(HarmonicBondedPotential):
    def __init__(self, *args, **kwargs):
        if 'range_' not in kwargs: kwargs['range_'] = (0,181)
        HarmonicBondedPotential.__init__(self, *args, **kwargs)

    @property
    def kscale_(self):
        return (180.0/np.pi)**2

    @property
    def type_(self):
        return 'angle'

class HarmonicDihedral(HarmonicBondedPotential):
    def __init__(self, *args, **kwargs):
        if 'range_' not in kwargs: kwargs['range_'] = (-180,180)
        HarmonicBondedPotential.__init__(self, *args, **kwargs)

    @property
    def kscale_(self):
        return (180.0/np.pi)**2

    @property
    def type_(self):
        return 'dihedral'

    @property
    def periodic(self):
        return True

class WLCSKPotential(HarmonicBondedPotential, metaclass=ABCMeta):
    """ ## https://aip.scitation.org/doi/full/10.1063/1.4968020 """
    def __init__(self, d, lp, kT, **kwargs):
        ## Note, we're leveraging the HarmonicBondedPotential base class and set k to lp here, but it isn't proper
        HarmonicBondedPotential.__init__(self, d, lp, **kwargs)
        self.d = d          # separation
        self.lp = lp            # persistence length
        self.kT = kT

    def filename(self,types=None):
        assert(types is None)
        return "%s%s-%.3f-%.3f.dat" % (self.filename_prefix, self.type_,
                                       self.d, self.lp)

    def __hash__(self):
        return hash((self.d, self.lp, self.kT, HarmonicBondedPotential.__hash__(self)))

    def __eq__(self, other):
        for a in ("d", "lp", "kT"):
            if self.__dict__[a] != other.__dict__[a]:
                return False
        return HarmonicBondedPotential.__eq__(self,other)
    
class WLCSKBond(WLCSKPotential):
    """ ## https://aip.scitation.org/doi/full/10.1063/1.4968020 """
    def __init__(self, d, lp, kT, **kwargs):
        if 'range_' not in kwargs: kwargs['range_'] = (0,50)
        if 'resolution' not in kwargs: kwargs['resolution'] = 0.02
        if 'max_force' not in kwargs: kwargs['max_force'] = 100
        WLCSKPotential.__init__(self, d, lp, kT, **kwargs)

    @property
    def type_(self):
        return "wlcbond"

    def potential(self, r, types=None):
        dr = r - self.d
        nk = self.d / (2*self.lp)
        q2 = (dr / self.d)**2
        a1,a2 = 1, -7.0/(2*nk)
        a3 = 3.0/32 - 3.0/(8*nk) - 6.0/(4*nk**2)
        p0,p1,p2,p3,p4 = 13.0/32, 3.4719,2.5064,-1.2906,0.6482
        a4 = (p0 + p1/(2*nk) + p2*(2*nk)**-2) / (1+p3/(2*nk)+p4*(2*nk)**-2)
        with np.errstate(divide='ignore',invalid='ignore'):
            u = self.kT * nk * ( a1/(1-q2) - a2*np.log(1-q2) + a3*q2 - 0.5*a4*q2*(q2-2) )
        return u

class WLCSKAngle(WLCSKPotential):
    """ ## https://aip.scitation.org/doi/full/10.1063/1.4968020 """
    def __init__(self, d, lp, kT, **kwargs):
        if 'range_' not in kwargs: kwargs['range_'] = (0,181)
        if 'resolution' not in kwargs: kwargs['resolution'] = 0.5
        WLCSKPotential.__init__(self, d, lp, kT, **kwargs)

    @property
    def type_(self):
        return "wlcangle"

    def potential(self, r, types=None):
        dr = r - self.d
        nk = self.d / (2*self.lp)
        p1,p2,p3,p4 = -1.237, 0.8105, -1.0243, 0.4595
        C = (1 + p1*(2*nk) + p2*(2*nk)**2) / (2*nk+p3*(2*nk)**2+p4*(2*nk)**3)
        u = self.kT * C * (1-np.cos(dr * np.pi / 180))
        return u

class NullPotential(AbstractPotential):
    def __init__(self, range_=(0,1), resolution=0.5, filename_prefix='./potentials/', *args, **kwargs):
        self.filename_prefix = filename_prefix
        AbstractPotential.__init__(self, range_=range_, resolution=resolution, *args,**kwargs)

    def potential(self, r, types):
        return np.zeros(r.shape)

    def filename(self, types=None):
        return f"{self.filename_prefix}nullpot.dat"
