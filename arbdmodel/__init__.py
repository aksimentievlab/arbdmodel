# -*- coding: utf-8 -*-
from pathlib import Path

from .version import get_version
__version__ = get_version() 

import numpy as np
from copy import copy, deepcopy
from inspect import ismethod
import os, sys, subprocess

_RESOURCE_DIR = Path(__file__).parent / 'resources'
def get_resource_path(relative_path):
    return _RESOURCE_DIR / relative_path

## Abstract classes
class Transformable():
    def __init__(self, position, orientation=None):
        self.position = np.array(position)
        if orientation is not None:
            orientation = np.array(orientation)
        self.orientation = orientation

    def transform(self, R = ((1,0,0),(0,1,0),(0,0,1)),
                  center = (0,0,0), offset = (0,0,0)):

        R,center,offset = [np.array(x) for x in (R,center,offset)]

        self.position = R.dot(self.position-center)+center+offset
                
        if self.orientation is not None:
            ## TODO: what if self.orientation is taken from parent?!
            self.orientation = self.orientation.dot(R)
        ...        

    def collapsedPosition(self):
        # print("collapsedPosition called", type(self), self.name)
        if isinstance(self, Child):
            # print(self.parent, isinstance(self.parent,Transformable))
            if isinstance(self.parent, Transformable):
                return self.applyOrientation(self.position) + self.parent.collapsedPosition()
            
                # if self.parent.orientation is not None:
                #     return self.parent.collapsedOrientation().dot(self.position) + self.parent.collapsedPosition()
        return np.array(self.position) # return a copy
                
    def applyOrientation(self,obj):
        # print("applyOrientation called", self.name, obj)
        if isinstance(self, Child):
            # print( self.orientation, self.orientation is not None, None is not None )
            # if self.orientation is not None:
            #     # print("applyOrientation applying", self, self.name, self.orientation)
            #     obj = self.orientation.dot(obj)
            if isinstance(self.parent, Transformable):
                if self.parent.orientation is not None:
                    obj = self.parent.orientation.dot(obj)
                obj = self.parent.applyOrientation(obj)
        # print("applyOrientation returning", self.name, obj)
        return obj

class Parent():
    def __init__(self, children=None, remove_duplicate_bonded_terms=False):
        self.children = []
        if children is not None:
            for x in children:
                self.add(x)
        
        self.remove_duplicate_bonded_terms = remove_duplicate_bonded_terms
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.impropers = []
        self.exclusions = []

        ## TODO: self.cacheInvalid = True # What will be in the cache?


    def add(self,x):
        ## TODO: check the parent-child tree to make sure there are no cycles
        if not isinstance(x,Child):
            raise Exception('Attempted to add an object to a group that does not inherit from the "Child" type')

        if x.parent is not None and x.parent is not self:
            raise Exception("Child {} already belongs to some group".format(x))
        x.parent = self
        self.children.append(x)

    def clear_all(self, keep_children=False):
        if keep_children == False:
            for x in self.children:
                x.parent = None
            self.children = []
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.impropers = []
        self.exclusions = []

    def remove(self,x):
        if x in self.children:
            self.children.remove(x)
            x.parent = None

    def add_bond(self, i,j, bond, exclude=False):
        ## TODO: how to handle duplicating and cloning bonds
        # beads = [b for b in self]
        # for b in (i,j): assert(b in beads)
        self.bonds.append( (i,j, bond, exclude) )

    def add_angle(self, i,j,k, angle):
        # beads = [b for b in self]
        # for b in (i,j,k): assert(b in beads)
        self.angles.append( (i,j,k, angle) )

    def add_dihedral(self, i,j,k,l, dihedral):
        # beads = [b for b in self]
        # for b in (i,j,k,l): assert(b in beads)
        self.dihedrals.append( (i,j,k,l, dihedral) )

    def add_improper(self, i,j,k,l, dihedral):
        # beads = [b for b in self]
        # for b in (i,j,k,l): assert(b in beads)
        self.impropers.append( (i,j,k,l, dihedral) )

    def add_exclusion(self, i,j):
        ## TODO: how to handle duplicating and cloning bonds
        ## TODO: perform following check elsewhere
        # beads = [b for b in self]
        # for b in (i,j): assert(b in beads)
        self.exclusions.append( (i,j) )

    def get_restraints(self):
        ret = []
        for c in self.children:
            ret.extend( c.get_restraints() )
        return ret

    def get_bonds(self):
        ret = self.bonds
        for c in self.children:
            if isinstance(c,Parent): ret.extend( c.get_bonds() )
        if self.remove_duplicate_bonded_terms:
            return list(set(ret))
        else:
            return ret


    def get_angles(self):
        ret = self.angles
        for c in self.children:
            if isinstance(c,Parent): ret.extend( c.get_angles() )
        if self.remove_duplicate_bonded_terms:
            return list(set(ret))
        else:
            return ret

    def get_dihedrals(self):
        ret = self.dihedrals
        for c in self.children:
            if isinstance(c,Parent): ret.extend( c.get_dihedrals() )
        if self.remove_duplicate_bonded_terms:
            return list(set(ret))
        else:
            return ret

    def get_impropers(self):
        ret = self.impropers
        for c in self.children:
            if isinstance(c,Parent): ret.extend( c.get_impropers() )
        if self.remove_duplicate_bonded_terms:
            return list(set(ret))
        else:
            return ret

    def get_exclusions(self):
        ret = self.exclusions
        for c in self.children:
            if isinstance(c,Parent): ret.extend( c.get_exclusions() )
        if self.remove_duplicate_bonded_terms:
            return list(set(ret))
        else:
            return ret

    ## Removed because prohibitively slow
    # def remove_duplicate_terms(self):
    #     for key in "bonds angles dihedrals impropers exclusions".split():
    #         self.remove_duplicate_item(key)

    # def remove_duplicate_item(self, dict_key, existing=None):
    #     if existing is None: existing = []
    #     ret = [i for i in list(set(self.__dict__[dict_key])) if i not in existing]
    #     self.__dict__[dict_key] = ret
    #     existing.extend(ret)
    #     for c in self.children:
    #         if isinstance(c,Parent): 
    #             ret = ret + c.remove_duplicate_item(dict_key, existing)
    #     return ret
        

    def __iter__(self):
        ## TODO: decide if this is the nicest way to do it!
        """Depth-first iteration through tree"""
        for x in self.children:
            if isinstance(x,Parent):
                if isinstance(x,Clone) and not isinstance(x.get_original_recursively(),Parent):
                    yield x
                else:
                    for y in x:
                        yield y
            else:
                yield x    

    def __len__(self):
        l = 0
        for x in self.children:
            if isinstance(x,Parent):
                l += len(x)
            else:
                l += 1
        return l
        
    def __getitem__(self, i):
        return self.children[i]
    
    def __setitem__(self, i, val):
        x = self.children[i]
        x.parent = None
        val.parent = self
        self.children[i] = val
        
class Child():
    def __init__(self, parent=None):
        self.parent = parent
        if parent is not None:
            assert( isinstance(parent, Parent) )
            parent.children.append(self)

    def __getattr__(self, name):
        """
        Try to get attribute from the parent
        """
        # if self.parent is not None:
        if "parent" not in self.__dict__ or self.__dict__["parent"] is None or name is "children":
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

        ## TODO: determine if there is a way to avoid __getattr__ if a method is being looked up  
        try:
            ret = getattr(self.parent,name)
        except:
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))
        if ismethod(ret):
            raise AttributeError("'{}' object has no method '{}'".format(type(self).__name__, name))
        return ret 

            
    # def __getstate__(self):
    #     print("Child getstate called", self)
    #     print(self.__dict__)
    #     return (self.__dict__,)

    # def __setstate__(self, state):
    #     self.__dict__, = state

class Clone(Transformable, Parent, Child):
    def __init__(self, original, parent=None,
                 position = None,
                 orientation = None):
        if position is None and original.position is not None:
            position = np.array( original.position )
        if orientation is None and original.orientation is not None:
            orientation = np.array( original.orientation )
        if parent is None:
            parent = original.parent
        self.original = original
        Child.__init__(self, parent)        
        Transformable.__init__(self, position, orientation)        

        ## TODO: keep own bond_list, etc, update when needed original changes

        if "children" in original.__dict__ and len(original.children) > 0:
            self.children = [Clone(c, parent = self) for c in original.children]
        else:
            self.children = []

    def get_original_recursively(self):
        if isinstance(self.original, Clone):
            return self.original.get_original_recursively()
        else:
            return self.original

    def __getattr__(self, name):
        """
        Try to get attribute from the original without descending the tree heirarchy, then look up parent

        TODO: handle PointParticle lookups into ParticleType
        """
        # print("Clone getattr",name)
        if name in self.original.__dict__:
            return self.original.__dict__[name]
        else:
            if "parent" not in self.__dict__ or self.__dict__["parent"] is None:
                raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))
            return getattr(self.parent, name)
        

## Particle classes
class ParticleType():
    """Class that hold common attributes that particles can point to"""

    excludedAttributes = ("idx","type_",
                          "position",
                          "children",
                          "parent", "excludedAttributes",
    )

    def __init__(self, name, charge=0, parent=None, **kargs):
        """ Parent type is used to fall back on for nonbonded interactions if this type is not specifically referenced """

        self.name        = name
        self.charge = charge
        self.parent = parent

        for key in ParticleType.excludedAttributes:
            assert( key not in kargs )

        for key,val in kargs.items():
            self.__dict__[key] = val

    def is_same_type(self, other):
        assert(isinstance(other,ParticleType))
        if self == other:
            return True
        elif self.parent is not None and self.parent == other:
            return True
        else:
            return False

    def __hash_key(self):
        l = [self.name,self.charge]
        for keyval in sorted(self.__dict__.items()):
            if isinstance(keyval[1], list): keyval = (keyval[0],tuple(keyval[1]))
            l.extend(keyval)
        return tuple(l)

    def __hash__(self):
        return hash(self.__hash_key())
    
    def _equal_check(a,b):
        if a.name == b.name:
            if a.__hash_key() != b.__hash_key():
                raise Exception("Two different ParticleTypes have same 'name' attribute")

    def __eq__(a,b):
        a._equal_check(b)
        return a.name == b.name
    def __lt__(a,b):
        a._equal_check(b)
        return a.name < b.name
    def __le__(a,b):
        a._equal_check(b)
        return a.name <= b.name
    def __gt__(a,b):
        a._equal_check(b)
        return a.name > b.name
    def __ge__(a,b):
        a._equal_check(b)
        return a.name >= b.name

    def __repr__(self):
        return '<{} {}{}>'.format( type(self), self.name, '[{}]'.format(self.parent) if self.parent is not None else '' )


class PointParticle(Transformable, Child):
    def __init__(self, type_, position, name="A", **kwargs):
        parent = None
        if 'parent' in kwargs:
            parent = kwargs['parent']
        Child.__init__(self, parent=parent)
        Transformable.__init__(self,position)

        self.type_    = type_                
        self.idx     = None
        self.name = name
        self.counter = 0
        self.restraints = []

        for key,val in kwargs.items():
            self.__dict__[key] = val
        
    def add_restraint(self, restraint):
        ## TODO: how to handle duplicating and cloning bonds
        self.restraints.append( restraint )

    def get_restraints(self):
        return [(self,r) for r in self.restraints]


    def __getattr__(self, name):
        """
        First try to get attribute from the parent, then type_
        
        Note that this data structure seems to be fragile, can result in stack overflow
        
        """
        # return Child.__getattr__(self,name)
        try:
            return Child.__getattr__(self,name)
        except Exception as e:
            if 'type_' in self.__dict__:
                return getattr(self.type_, name)
            else:
                raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def _get_psfpdb_dictionary(self):
        p = self
        try:
            segname = p.segname
        except:
            segname = "A"
        try:
            chain = p.chain
        except:
            chain = "A"
        try:
            resname = p.resname
        except:
            resname = p.name[:3]
        try:
            resid = p.resid
        except:
            resid = p.idx+1

        try:
            occ = p.occupancy
        except:
            occ = 0
        try:
            beta = p.beta
        except:
            beta = 0

        data = dict(segname = segname,
                    resname = resname,
                    name = str(p.name)[:4],
                    chain = chain[0],
                    resid = int(resid),
                    idx = p.idx+1,
                    type = p.type_.name[:4],
                    charge = p.charge,
                    mass = p.mass,
                    occupancy = occ,
                    beta = beta
                )
        return data


class Group(Transformable, Parent, Child):

    def __init__(self, name=None, children = None, parent=None, 
                 position = np.array((0,0,0)),
                 orientation = np.array(((1,0,0),(0,1,0),(0,0,1))),
                 remove_duplicate_bonded_terms = False,
                 **kwargs):

        Transformable.__init__(self, position, orientation)
        Child.__init__(self, parent) # Initialize Child first
        Parent.__init__(self, children, remove_duplicate_bonded_terms)
        self.name = name
        self.isClone = False

        for key,val in kwargs.items():
            self.__dict__[key] = val


    def clone(self):
        return Clone(self)
        g = copy(self)
        g.isClone = True        # TODO: use?
        g.children = [copy(c) for c in g.children]
        for c in g.children:
            c.parent = g
        return g
        g = Group(position = self.position,
                  orientation = self.orientation)
        g.children = self.children # lists point to the same object

    def duplicate(self):
        new = deepcopy(self)
        for c in new.children:
            c.parent = new
        return new
        # Group(position = self.position,
        #       orientation = self.orientation)
        # g.children = deepcopy self.children.deepcopy() # lists are the same object

    ## TODO override deepcopy so parent can be excluded from copying?
        
    # def __getstate__(self):
    #     return (self.children, self.parent, self.position, self.orientation)

    # def __setstate__(self, state):
    #     self.children, self.parent, self.position, self.orientation = state

        
class PdbModel(Transformable, Parent):

    def __init__(self, children=None, dimensions=None, remove_duplicate_bonded_terms=False):
        Transformable.__init__(self,(0,0,0))
        Parent.__init__(self, children, remove_duplicate_bonded_terms)
        self.dimensions = dimensions
        self.particles = [p for p in self]
        self.cacheInvalid = True

    def _updateParticleOrder(self):
        pass

    def writePdb(self, filename, beta_from_fixed=False):
        if self.cacheInvalid:
            self._updateParticleOrder()
        with open(filename,'w') as fh:
            ## Write header
            fh.write("CRYST1{:>9.3f}{:>9.3f}{:>9.3f}  90.00  90.00  90.00 P 1           1\n".format( *self.dimensions ))

            ## Write coordinates
            formatString = "ATOM {idx:>6.6s} {name:^4.4s} {resname:3.3s} {chain:1.1s}{resid:>5.5s}   {x:8.8s}{y:8.8s}{z:8.8s}{occupancy:6.2f}{beta:6.2f}  {charge:2d}{segname:>6s}\n"
            for p in self.particles:
                ## http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
                data = p._get_psfpdb_dictionary()
                idx = data['idx']

                if np.log10(idx) >= 5:
                    idx = " *****"
                else:
                    idx = "{:>6d}".format(idx)
                data['idx'] = idx

                if beta_from_fixed:
                    data['beta'] = 1 if 'fixed' in p.__dict__ else 0

                pos = p.collapsedPosition()
                dig = [max(int(np.log10(np.abs(x)+1e-6)//1),0)+1 for x in pos]
                for d in dig: assert( d <= 7 )
                # assert( np.all(dig <= 7) )
                fs = ["{: %d.%df}" % (8,7-d) for d in dig]
                x,y,z = [f.format(x) for f,x in zip(fs,pos)] 
                data['x'] = x
                data['y'] = y
                data['z'] = z
                assert(data['resid'] < 1e5)
                data['charge'] = int(data['charge'])
                data['resid'] = "{:<4d}".format(data['resid'])
                fh.write( formatString.format(**data) )

        return
        
    def writePsf(self, filename):
        if self.cacheUpToDate == False:
            self._updateParticleOrder()
        with open(filename,'w') as fh:
            ## Write header
            fh.write("PSF NAMD\n\n") # create NAMD formatted psf
            fh.write("{:>8d} !NTITLE\n\n".format(0))
            
            ## ATOMS section
            fh.write("{:>8d} !NATOM\n".format(len(self.particles)))

            ## From vmd/plugins/molfile_plugin/src/psfplugin.c
            ## "%d %7s %10s %7s %7s %7s %f %f"
            formatString = "{idx:>8d} {segname:7.7s} {resid:<10.10s} {resname:7.7s}" + \
                           " {name:7.7s} {type:7.7s} {charge:f} {mass:f}\n"
            for p in self.particles:
                data = p._get_psfpdb_dictionary()
                data['resid'] = "%d%c%c" % (data['resid']," "," ") # TODO: work with large indices
                fh.write( formatString.format(**data) )
            fh.write("\n")

            ## Write out bonds
            bonds = self.get_bonds()
            fh.write("{:>8d} !NBOND\n".format(len(bonds)))
            counter = 0
            for p1,p2,b,ex in bonds:
                fh.write( "{:>8d}{:>8d}".format(p1.idx+1,p2.idx+1) )
                counter += 1
                if counter == 4:
                    fh.write("\n")
                    counter = 0
                else:
                    fh.write(" ")
            fh.write("\n" if counter == 0 else "\n\n")

            ## Write out angles
            angles = self.get_angles()
            fh.write("{:>8d} !NTHETA\n".format(len(angles)))
            counter = 0
            for p1,p2,p3,a in angles:
                fh.write( "{:>8d}{:>8d}{:>8d}".format(p1.idx+1,p2.idx+1,p3.idx+1) )
                counter += 1
                if counter == 3:
                    fh.write("\n")
                    counter = 0
                else:
                    fh.write(" ")
            fh.write("\n" if counter == 0 else "\n\n")

            ## Write out dihedrals
            dihedrals = self.get_dihedrals()
            fh.write("{:>8d} !NPHI\n".format(len(dihedrals)))
            counter = 0
            for p1,p2,p3,p4,a in dihedrals:
                fh.write( "{:>8d}{:>8d}{:>8d}{:>8d}".format(p1.idx+1,p2.idx+1,p3.idx+1,p4.idx+1) )
                counter += 1
                if counter == 2:
                    fh.write("\n")
                    counter = 0
                else:
                    fh.write(" ") 
            fh.write("\n" if counter == 0 else "\n\n")

            ## Write out impropers
            impropers = self.get_impropers()
            fh.write("{:>8d} !NIMPHI\n".format(len(impropers)))
            counter = 0
            for p1,p2,p3,p4,a in impropers:
                fh.write( "{:>8d}{:>8d}{:>8d}{:>8d}".format(p1.idx+1,p2.idx+1,p3.idx+1,p4.idx+1) )
                counter += 1
                if counter == 2:
                    fh.write("\n")
                    counter = 0
                else:
                    fh.write(" ")
            fh.write("\n" if counter == 0 else "\n\n")

            fh.write("\n{:>8d} !NDON: donors\n\n\n".format(0))
            fh.write("\n{:>8d} !NACC: acceptors\n\n\n".format(0))
            fh.write("\n       0 !NNB\n\n")
            natoms = len(self.particles)
            for i in range(natoms//8):
                fh.write("      0       0       0       0       0       0       0       0\n")
            for i in range(natoms-8*(natoms//8)):
                fh.write("      0")
            fh.write("\n\n       1       0 !NGRP\n\n")


class ArbdModel(PdbModel):
    def __init__(self, children, dimensions=(1000,1000,1000), temperature=291, timestep=50e-6,
                 particle_integrator = 'Brown',
                 cutoff=50, decompPeriod=10000, pairlistDistance=None, nonbondedResolution=0.1,
                 remove_duplicate_bonded_terms=True):

        PdbModel.__init__(self, children, dimensions, remove_duplicate_bonded_terms)
        self.temperature = temperature

        self.timestep = timestep
        self.cutoff  =  cutoff

        self.particle_integrator = particle_integrator
        
        if pairlistDistance == None:
            pairlistDistance = cutoff+10
        
        self.decompPeriod = decompPeriod
        self.pairlistDistance = pairlistDistance

        self.numParticles = 0
        self.particles = []
        self.type_counts = None

        self.nbSchemes = []
        self._nbParamFiles = [] # This could be made more robust
        self.nbResolution = 0.1

        self._written_bond_files = dict()        

        self.cacheUpToDate = False

    def clear_all(self, keep_children=False):
        Parent.clear_all(self, keep_children=keep_children)
        self.particles = []
        self.numParticles = 0
        self.type_counts = None
        self._nbParamFiles = []
        self._written_bond_files = dict()

    def _getNbScheme(self, typeA, typeB):
        scheme = None
        for s,A,B in self.nbSchemes:
            if A is None or B is None:
                if A is None and B is None:
                    return s
                elif A is None and typeB.is_same_type(B):
                    return s
                elif B is None and typeA.is_same_type(A):
                    return s
            elif typeA.is_same_type(A) and typeB.is_same_type(B):
                return s
        raise Exception("No nonbonded scheme found for %s and %s" % (typeA.name, typeB.name))

    def _countParticleTypes(self):
        ## TODO: check for modifications to particle that require
        ## automatic generation of new particle type
        type_counts = dict()
        for p in self:
            t = p.type_
            if t in type_counts:
                type_counts[t] += 1
            else:
                type_counts[t] = 1
        self.type_counts = type_counts

    def _updateParticleOrder(self):
        ## Create ordered list
        self.particles = [p for p in self]
        # self.particles = sorted(particles, key=lambda p: (p.type_, p.idx))
        
        ## Update particle indices
        for p,i in zip(self.particles,range(len(self.particles))):
            p.idx = i
            
        # self.initialCoords = np.array([p.initialPosition for p in self.particles])

    def useNonbondedScheme(self, nbScheme, typeA=None, typeB=None):
        self.nbSchemes.append( (nbScheme, typeA, typeB) )
        if typeA != typeB:
            self.nbSchemes.append( (nbScheme, typeB, typeA) )

    def simulate(self, output_name, output_directory='output', num_steps=100000000, timestep=None, gpu=0, output_period=1e4, arbd=None, directory='.', replicas=1):
        assert(type(gpu) is int)
        num_steps = int(num_steps)

        d_orig = os.getcwd()
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
            os.chdir(directory)

            if timestep is not None:
                self.timestep = timestep

            if self.cacheUpToDate == False: # TODO: remove cache?
                self._countParticleTypes()
                self._updateParticleOrder()

            if output_directory == '': output_directory='.'

            if arbd is None:
                for path in os.environ["PATH"].split(os.pathsep):
                    path = path.strip('"')
                    fname = os.path.join(path, "arbd")
                    if os.path.isfile(fname) and os.access(fname, os.X_OK):
                        arbd = fname
                        break 

            if arbd is None: raise Exception("ARBD was not found")

            if not os.path.exists(arbd):
                raise Exception("ARBD was not found")
            if not os.path.isfile(arbd):
                raise Exception("ARBD was not found")
            if not os.access(arbd, os.X_OK):
                raise Exception("ARBD is not executable")

            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            elif not os.path.isdir(output_directory):
                raise Exception("output_directory '%s' is not a directory!" % output_directory)


            self.writePdb( output_name + ".pdb" )
            self.writePsf( output_name + ".psf" )
            self.writeArbdFiles( output_name, numSteps=num_steps, outputPeriod=output_period )
            os.sync()

            ## http://stackoverflow.com/questions/18421757/live-output-from-subprocess-command

            cmd = [arbd, '-g', "%d" % gpu]
            if replicas > 1:
                cmd = cmd + ['-r',replicas]
            cmd = cmd + ["%s.bd" % output_name, "%s/%s" % (output_directory, output_name)]
            cmd = tuple(str(x) for x in cmd)

            print("Running ARBD with: %s" % " ".join(cmd))
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()

        except:
            raise
        finally:
            os.chdir(d_orig)


    # -------------------------- #
    # Methods for printing model #
    # -------------------------- #

    def writeArbdFiles(self, prefix, numSteps=100000000, outputPeriod=10000):
        ## TODO: save and reference directories and prefixes using member data
        d = self.potential_directory = "potentials"
        if not os.path.exists(d):
            os.makedirs(d)
        self._restraint_filename = "%s/%s.restraint.txt" % (d, prefix)
        self._bond_filename = "%s/%s.bonds.txt" % (d, prefix)
        self._angle_filename = "%s/%s.angles.txt" % (d, prefix)
        self._dihedral_filename = "%s/%s.dihedrals.txt" % (d, prefix)
        self._exclusion_filename = "%s/%s.exculsions.txt" % (d, prefix)
        
        # self._writeArbdCoordFile( prefix + ".coord.txt" )
        self._writeArbdParticleFile( prefix + ".particles.txt" )
        self._writeArbdRestraintFile()
        self._writeArbdBondFile()
        self._writeArbdAngleFile()
        self._writeArbdDihedralFile()
        self._writeArbdExclusionFile()
        self._writeArbdPotentialFiles( prefix, directory = d )
        self._writeArbdConf( prefix, numSteps=numSteps, outputPeriod=outputPeriod )
        
    # def _writeArbdCoordFile(self, filename):
    #     with open(filename,'w') as fh:
    #         for p in self.particles:
    #             fh.write("%f %f %f\n" % tuple(x for x in p.collapsedPosition()))

    def _writeArbdParticleFile(self, filename):
        with open(filename,'w') as fh:
            if self.particle_integrator == "Brown":
                for p in self.particles:
                    data = tuple([p.idx,p.type_.name] + [x for x in p.collapsedPosition()])
                    fh.write("ATOM %d %s %f %f %f\n" % data)
            else:
                for p in self.particles:
                    data = [p.idx,p.type_.name] + [x for x in p.collapsedPosition()]
                    try:
                        data = data + p.momentum
                    except:
                        try:
                            data = data + p.velocity*p.mass
                        except:
                            data = data + [0,0,0]
                    fh.write("ATOM %d %s %f %f %f %f %f %f\n" % tuple(data))
                

        
    def _writeArbdConf(self, prefix, randomSeed=None, numSteps=100000000, outputPeriod=10000, restartCoordinateFile=None):
        ## TODO: raise exception if _writeArbdPotentialFiles has not been called
        filename = "%s.bd" % prefix

        ## Prepare a dictionary to fill in placeholders in the configuration file
        params = self.__dict__.copy() # get parameters from System object

        if randomSeed is None:
            params['randomSeed']     = ""
        else:
            params['randomSeed'] = "seed %s" % randomSeed
        params['numSteps']       = int(numSteps)

        # params['coordinateFile'] = "%s.coord.txt" % prefix
        params['particleFile'] = "%s.particles.txt" % prefix
        if restartCoordinateFile is None:
            params['restartCoordinates'] = ""
        else:
            params['restartCoordinates'] = "restartCoordinates %s" % restartCoordinateFile
        params['outputPeriod'] = outputPeriod

        for k,v in zip('XYZ', self.dimensions):
            params['origin'+k] = -v*0.5
            params['dim'+k] = v
        
        params['pairlistDistance'] -= params['cutoff'] 

        ## Actually write the file
        with open(filename,'w') as fh:
            fh.write("""{randomSeed}
timestep {timestep}
steps {numSteps}
numberFluct 0                   # deprecated

interparticleForce 1            # other values deprecated
fullLongRange 0                 # deprecated
temperature {temperature}
ParticleDynamicType {particle_integrator}

outputPeriod {outputPeriod}
## Energy doesn't actually get printed!
outputEnergyPeriod {outputPeriod}
outputFormat dcd

## Infrequent domain decomposition because this kernel is still very slow
decompPeriod {decompPeriod}
cutoff {cutoff}
pairlistDistance {pairlistDistance}

origin {originX} {originY} {originZ}
systemSize {dimX} {dimY} {dimZ}
\n""".format(**params))
            
            ## Write entries for each type of particle
            for pt,num in self.getParticleTypesAndCounts():
                ## TODO create new particle types if existing has grid
                particleParams = pt.__dict__.copy()
                particleParams['num'] = num
                if self.particle_integrator in ('Brown','Brownian'):
                    try:
                        D = pt.diffusivity
                    except:
                        """ units "k K/(amu/ns)" "AA**2/ns" """
                        D = 831447.2 * self.temperature / (pt.mass * pt.damping_coefficient)
                    particleParams['dynamics'] = 'diffusion {D}'.format(D = D)
                elif self.particle_integrator == 'Langevin':
                    try:
                        gamma = pt.damping_coefficient
                    except:
                        """ units "k K/(AA**2/ns)" "amu/ns" """
                        gamma = 831447.2 * self.temperature / (pt.mass*pt.diffusivity)
                    particleParams['dynamics'] = """mass {mass}
transDamping {g} {g} {g}
""".format(mass=pt.mass, g=gamma)
                else:
                    raise ValueError("Unrecognized particle integrator '{}'".format(self.particle_integrator))
                fh.write("""
particle {name}
num {num}
{dynamics}
""".format(**particleParams))
                if 'grid' in particleParams:
                    if not isinstance(pt.grid, list): pt.grid = [pt.grid]
                    for g,s in pt.grid:
                        ## TODO, use Path.relative_to?
                        try:
                            fh.write("gridFile {}\n".format(g.relative_to(os.getcwd())))
                        except:
                            fh.write("gridFile {}\n".format(g))

                        fh.write("gridFileScale {}\n".format(s))

                else:
                    fh.write("gridFile {}/null.dx\n".format(self.potential_directory))

            ## Write coordinates and interactions
            fh.write("""
## Input coordinates
inputParticles {particleFile}
{restartCoordinates}

## Interaction potentials
tabulatedPotential  1
## The i@j@file syntax means particle type i will have NB interactions with particle type j using the potential in file
""".format(**params))
            for pair,f in zip(self._particleTypePairIter(), self._nbParamFiles):
                i,j,t1,t2 = pair
                fh.write("tabulatedFile %d@%d@%s\n" % (i,j,f))

            ## Bonded interactions
            restraints = self.get_restraints()
            bonds = self.get_bonds()
            angles = self.get_angles()
            dihedrals = self.get_dihedrals()
            exclusions = self.get_exclusions()

            if len(bonds) > 0:
                for b in list(set([b for i,j,b,ex in bonds])):
                    fh.write("tabulatedBondFile %s\n" % b)

            if len(angles) > 0:
                for b in list(set([b for i,j,k,b in angles])):
                    fh.write("tabulatedAngleFile %s\n" % b)

            if len(dihedrals) > 0:
                for b in list(set([b for i,j,k,l,b in dihedrals])):
                    fh.write("tabulatedDihedralFile %s\n" % b)

            if len(restraints) > 0:
                fh.write("inputRestraints %s\n" % self._restraint_filename)
            if len(bonds) > 0:
                fh.write("inputBonds %s\n" % self._bond_filename)
            if len(angles) > 0:
                fh.write("inputAngles %s\n" % self._angle_filename)
            if len(dihedrals) > 0:
                fh.write("inputDihedrals %s\n" % self._dihedral_filename)
            if len(exclusions) > 0:
                fh.write("inputExcludes %s\n" % self._exclusion_filename)
     
        write_null_dx = False
        for pt,num in self.getParticleTypesAndCounts():
            if "grid" not in pt.__dict__: 
                gridfile = "{}/null.dx".format(self.potential_directory)
                with open(gridfile, 'w') as fh:
                    fh.write("""object 1 class gridpositions counts  2 2 2
origin {originX} {originY} {originZ}
delta  {dimX} 0.000000 0.000000
delta  0.000000 {dimY} 0.000000
delta  0.000000 0.000000 {dimZ}
object 2 class gridconnections counts  2 2 2
object 3 class array type float rank 0 items 8 data follows
0.0	0.0	0.0	
0.0	0.0	0.0	
0.0	0.0	
attribute "dep" string "positions"
object "density" class field 
component "positions" value 1
component "connections" value 2
component "data" value 3
""".format(**params))
                    break

    def getParticleTypesAndCounts(self):
        ## TODO: remove(?)
        return sorted( self.type_counts.items(), key=lambda x: x[0] )

    def _particleTypePairIter(self):
        typesAndCounts = self.getParticleTypesAndCounts()
        for i in range(len(typesAndCounts)):
            t1 = typesAndCounts[i][0]
            for j in range(i,len(typesAndCounts)):
                t2 = typesAndCounts[j][0]
                yield( (i,j,t1,t2) )
    
    def _writeArbdPotentialFiles(self, prefix, directory = "potentials"):
        try: 
            os.makedirs(directory)
        except OSError:
            if not os.path.isdir(directory):
                raise

        pathPrefix = "%s/%s" % (directory,prefix)
        self._writeNonbondedParameterFiles( pathPrefix + "-nb" )
        # self._writeBondParameterFiles( pathPrefix )
        # self._writeAngleParameterFiles( pathPrefix )
        # self._writeDihedralParameterFiles( pathPrefix )
                
    def _writeNonbondedParameterFiles(self, prefix):
        x = np.arange(0, self.cutoff, self.nbResolution)
        for i,j,t1,t2 in self._particleTypePairIter():
            f = "%s.%s-%s.dat" % (prefix, t1.name, t2.name)
            scheme = self._getNbScheme(t1,t2)
            scheme.write_file(f, t1, t2, rMax = self.cutoff)
            self._nbParamFiles.append(f)

    def _getNonbondedPotential(self,x,a,b):
        return a*(np.exp(-x/b))    

    def _writeArbdRestraintFile( self ):
        with open(self._restraint_filename,'w') as fh:
            for i,restraint in self.get_restraints():
                item = [i.idx]
                if len(restraint) == 1:
                    item.append(restraint[0])
                    item.extend(i.get_collapsed_position())
                elif len(restraint) == 2:
                    item.append(restraint[0])
                    item.extend(restraint[1])
                elif len(restraint) == 5:
                    item.extend(restraint)
                fh.write("RESTRAINT %d %f %f %f %f\n" % tuple(item))

    def _writeArbdBondFile( self ):
        for b in list( set( [b for i,j,b,ex in self.get_bonds()] ) ):
            if type(b) is not str and not isinstance(b, Path):
                b.write_file()

        with open(self._bond_filename,'w') as fh:
            for i,j,b,ex in self.get_bonds():
                item = (i.idx, j.idx, str(b))
                if ex:
                    fh.write("BOND REPLACE %d %d %s\n" % item)
                else:
                    fh.write("BOND ADD %d %d %s\n" % item)

    def _writeArbdAngleFile( self ):
        for b in list( set( [b for i,j,k,b in self.get_angles()] ) ):
            if type(b) is not str and not isinstance(b, Path):
                b.write_file()

        with open(self._angle_filename,'w') as fh:
            for b in self.get_angles():
                item = tuple([p.idx for p in b[:-1]] + [str(b[-1])])
                fh.write("ANGLE %d %d %d %s\n" % item)

    def _writeArbdDihedralFile( self ):
        for b in list( set( [b for i,j,k,l,b in self.get_dihedrals()] ) ):
            if type(b) is not str and not isinstance(b, Path):
                b.write_file()

        with open(self._dihedral_filename,'w') as fh:
            for b in self.get_dihedrals():
                item = tuple([p.idx for p in b[:-1]] + [str(b[-1])])
                fh.write("DIHEDRAL %d %d %d %d %s\n" % item)

    def _writeArbdExclusionFile( self ):
        with open(self._exclusion_filename,'w') as fh:
            for ex in self.get_exclusions():
                item = tuple(int(p.idx) for p in ex)
                fh.write("EXCLUDE %d %d\n" % item)

    def dimensions_from_structure( self, padding_factor=1.5, isotropic=False ):
        ## TODO: cache coordinates using numpy arrays for quick min/max
        raise(NotImplementedError)

    def write_namd_configuration( self, output_name, num_steps = 1e6,
                                  output_directory = 'output',
                                  update_dimensions=True, extrabonds=True ):

        format_data = self.__dict__.copy() # get parameters from System object

        format_data['extrabonds'] = """extraBonds on
extraBondsFile $prefix.exb
""" if extrabonds else ""

        if self.useTclForces:
            format_data['margin'] = ""
            format_data['tcl_forces'] = """tclForces on
tclForcesScript $prefix.forces.tcl
"""
        else:
            format_data['margin'] = """margin              30
"""
            format_data['tcl_forces'] = ""

        if update_dimensions:
            format_data['dimensions'] = self.dimensions_from_structure()

        for k,v in zip('XYZ', format_data['dimensions']):
            format_data['origin'+k] = -v*0.5
            format_data['cell'+k] = v

        format_data['prefix'] = output_name
        format_data['num_steps'] = int(num_steps//12)*12
        format_data['output_directory'] = output_directory
        filename = '{}.namd'.format(output_name)

        with open(filename,'w') as fh:
            fh.write("""
set prefix {prefix}
set nLast 0;			# increment when continueing a simulation
set n [expr $nLast+1]
set out {output_directory}/$prefix-$n
set temperature {temperature}

structure          $prefix.psf
coordinates        $prefix.pdb

outputName         $out
XSTfile            $out.xst
DCDfile            $out.dcd

#############################################################
## SIMULATION PARAMETERS                                   ##
#############################################################

# Input
paraTypeCharmm	    on
parameters          charmm36.nbfix/par_all36_na.prm
parameters	    charmm36.nbfix/par_water_ions_na.prm

wrapAll             off

# Force-Field Parameters
exclude             scaled1-4
1-4scaling          1.0
switching           on
switchdist           8
cutoff              10
pairlistdist        12
{margin}

# Integrator Parameters
timestep            2.0  ;# 2fs/step
rigidBonds          all  ;# needed for 2fs steps
nonbondedFreq       1
fullElectFrequency  3
stepspercycle       12

# PME (for full-system periodic electrostatics)
PME                 no
PMEGridSpacing      1.2

# Constant Temperature Control
langevin            on    ;# do langevin dynamics
# langevinDamping     1   ;# damping coefficient (gamma); used in original study
langevinDamping     0.1   ;# less friction for faster relaxation
langevinTemp        $temperature
langevinHydrogen    off    ;# don't couple langevin bath to hydrogens

# output
useGroupPressure    yes
xstFreq             4800
outputEnergies      4800
dcdfreq             4800
restartfreq         48000

#############################################################
## EXTRA FORCES                                            ##
#############################################################

# ENM and intrahelical extrabonds
{extrabonds}
{tcl_forces}

#############################################################
## RUN                                                     ##
#############################################################

# Continuing a job from the restart files
cellBasisVector1 {cellX} 0 0
cellBasisVector2 0 {cellY} 0
cellBasisVector3 0 0 {cellZ}

if {{$nLast == 0}} {{
    temperature 300
    fixedAtoms on
    fixedAtomsForces on
    fixedAtomsFile $prefix.fixed.pdb
    fixedAtomsCol B
    minimize 2400
    fixedAtoms off
    minimize 2400
}} else {{
    bincoordinates  {output_directory}/$prefix-$nLast.restart.coor
    binvelocities   {output_directory}/$prefix-$nLast.restart.vel
}}

run {num_steps:d}
""".format(**format_data))

    def atomic_simulate(self, output_name, output_directory='output'):
        if self.cacheUpToDate == False: # TODO: remove cache?
            self._countParticleTypes()
            self._updateParticleOrder()

        if output_directory == '': output_directory='.'
        self.writePdb( output_name + ".pdb" )
        self.writePdb( output_name + ".fixed.pdb", beta_from_fixed=True )
        self.writePsf( output_name + ".psf" )
        self.write_namd_configuration( output_name, output_directory = output_directory )
        os.sync()

