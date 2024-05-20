# -*- coding: utf-8 -*-

## Set up loggers
import logging
def _get_username():
    import sys
    try:
        return sys.environ['USER']
    except:
        return None

logging.basicConfig(format='%(name)s: %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_ch = logging.StreamHandler()
_ch.setFormatter(logging.Formatter('%(name)s: %(levelname)s: %(message)s'))
logger.addHandler(_ch)
logger.propagate = False

devlogger = logging.getLogger(__name__+'.dev')
# devlogger.setLevel(logging.DEBUG)
if _get_username() not in ('cmaffeo2',):
    devlogger.addHandler(logging.NullHandler())

## Import resources
from pathlib import Path
from abc import abstractmethod, ABCMeta

from .version import get_version
__version__ = get_version() 

import numpy as np
from copy import copy, deepcopy
from inspect import ismethod
import os, sys, subprocess
import shutil


_RESOURCE_DIR = Path(__file__).parent / 'resources'
def get_resource_path(relative_path):
    return _RESOURCE_DIR / relative_path

def _get_properties_and_dict_keys(obj):
    import inspect
    cls = obj.__class__

    def filter_props(name_type):
        nt = name_type
        return not nt[0].startswith('_') and isinstance(nt[1],property)
    properties = [name for name,type_ in filter(filter_props, inspect.getmembers(obj.__class__))]
    return properties + list(obj.__dict__.keys())

__active_model = None           # variable to be set temporarily by models as they are doing something, used by AbstractPotential classes

## Abstract classes
class Transformable():
    def __init__(self, position, orientation=None):
        self.position = np.array(position)
        if orientation is not None:
            orientation = np.array(orientation)
        self.orientation = orientation

    def translate(self, offset = (0,0,0)):
        self.transform( offset = offset )

    def rotate(self, R, about = (0,0,0)):
        self.transform( R = R, center = about )

    def transform(self, R = ((1,0,0),(0,1,0),(0,0,1)),
                  center = (0,0,0), offset = (0,0,0)):

        R,center,offset = [np.array(x) for x in (R,center,offset)]

        self.position = R.dot(self.position-center)+center+offset
                
        if self.orientation is not None:
            ## TODO: what if self.orientation is taken from parent?!
            self.orientation = self.orientation.dot(R)
        ...        

    def get_collapsed_position(self):
        # print("get_collapsed_position called", type(self), self.name)
        if isinstance(self, Child):
            # print(self.parent, isinstance(self.parent,Transformable))
            if isinstance(self.parent, Transformable):
                return self.applyOrientation(self.position) + self.parent.get_collapsed_position()
            
                # if self.parent.orientation is not None:
                #     return self.parent.collapsedOrientation().dot(self.position) + self.parent.get_collapsed_position()
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
        self.bond_angles = []
        self.product_potentials = []
        self.group_sites = []
        
        self.rigid = False
        ## TODO: self.cacheInvalid = True # What will be in the cache?

    def add(self,x):
        ## TODO: check the parent-child tree to make sure there are no cycles
        if not isinstance(x,Child):
            raise Exception('Attempted to add an object to a group that does not inherit from the "Child" type')

        if x.parent is not None and x.parent is not self:
            raise Exception("Child {} already belongs to some group".format(x))
        x.parent = self
        self.children.append(x)

    def insert(self,idx,x):
        ## TODO: check the parent-child tree to make sure there are no cycles
        if not isinstance(x,Child):
            raise Exception('Attempted to add an object to a group that does not inherit from the "Child" type')

        if x.parent is not None and x.parent is not self:
            raise Exception("Child {} already belongs to some group".format(x))
        x.parent = self
        self.children.insert(idx,x)

    def index(self, x):
        return self.children.index(x)

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
        self.bond_angles = []
        self.product_potentials = []
        self.group_sites = []

    def remove(self,x):
        if x in self.children:
            self.children.remove(x)
            if x.parent is self:
                x.parent = None

    def get_center(self, weight=None):
        if weight is None:
            center = np.mean([p.get_collapsed_position() for p in self], axis=0)
        elif weight == 'mass':
            raise NotImplementedError('')
        return center

    def add_bond(self, i,j, bond, exclude=False):
        assert( i is not j )
        ## TODO: how to handle duplicating and cloning bonds
        # beads = [b for b in self]
        # for b in (i,j): assert(b in beads)
        self.bonds.append( (i,j, bond, exclude) )

    def add_angle(self, i,j,k, angle):
        assert( len(set((i,j,k))) == 3 )
        # beads = [b for b in self]
        # for b in (i,j,k): assert(b in beads)
        self.angles.append( (i,j,k, angle) )

    def add_dihedral(self, i,j,k,l, dihedral):
        assert( len(set((i,j,k,l))) == 4 )

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

    def add_bond_angle(self, i,j,k,l, bond_angle, exclude=False):
        assert( len(set((i,j,k,l))) == 4 )
        ## TODO: how to handle duplicating and cloning bonds
        # beads = [b for b in self]
        # for b in (i,j): assert(b in beads)
        self.bond_angles.append( (i,j,k,l, bond_angle) )

    def add_product_potential(self, potential_list):
        """ potential_list: list of tuples of form (particle_i, particle_j,..., TabulatedPotential) """
        if len(potential_list) < 2: raise ValueError("Too few potentials")
        for elem in potential_list:
            beads = elem[:-1]
            pot = elem[-1]
            if len(beads) < 2: raise ValueError("Too few particles specified in product_potential")
            if len(beads) > 4: raise ValueError("Too many particles specified in product_potential")

        self.product_potentials.append(potential_list)
        ## TODO: how to handle duplicating and cloning bonds

    def get_restraints(self):
        ret = []
        for c in self.children +  self.group_sites:
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

    def get_bond_angles(self):
        ret = self.bond_angles
        for c in self.children:
            if isinstance(c,Parent): ret.extend( c.get_bond_angles() )
        if self.remove_duplicate_bonded_terms:
            return list(set(ret))
        else:
            return ret

    def get_product_potentials(self):
        ret = self.product_potentials
        for c in self.children:
            if isinstance(c,Parent): ret.extend( c.get_product_potentials() )
        if self.remove_duplicate_bonded_terms:
            return list(set(ret))
        else:
            return ret

    def _get_bond_potentials(self):
        bonds =  [b for i,j,b,ex in self.get_bonds()]
        bondangles1 = [b[1] for i,j,k,l,b in self.get_bond_angles()]
        return list(set( bonds+bondangles1 ))

    def _get_angle_potentials(self):
        angles =  [b for i,j,k,b in self.get_angles()]
        bondangles1 = [b[0] for i,j,k,l,b in self.get_bond_angles()]
        bondangles2 = [b[2] for i,j,k,l,b in self.get_bond_angles()]
        return list(set( angles+bondangles1+bondangles2 ))


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
        # devlogger.info(f'{self}.__iter__(): 0th child {None if len(self.children) == 0 else self.children[0]}')
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
        if "parent" not in self.__dict__ or self.__dict__["parent"] is None or name == "children":
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))


        excluded_attributes = ['parent','rigid']
        if name in excluded_attributes:
            raise AttributeError("'{}' object has no attribute '{}' and cannot look it up from the parent".format(type(self).__name__, name))

        ## TODO: determine if there is a way to avoid __getattr__ if a method is being looked up  
        try:
            ret = getattr(self.parent,name)
        except:
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))
        if ismethod(ret):
            raise AttributeError("'{}' object has no method '{}'".format(type(self).__name__, name))
        return ret 

    def _clear_types(self):
        if self.parent is not None:
            self.parent._clear_types()
            
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
                          "orientation",
                          "children",
                          "name",
                          "parent", "excludedAttributes",
    )

    def __init__(self, name, charge=0, parent=None, rigid=False, rigid_body_potential=None, **kwargs):
        """ Parent type is used to fall back on for nonbonded interactions if this type is not specifically referenced """

        if parent is not None:
            for k,v in parent.__dict__.items():
                if k not in ParticleType.excludedAttributes:
                    self.__dict__[k] = v

        self.name   = name
        self.charge = charge
        self.parent = parent
        self.rigid = rigid            
        self.rigid_body_potential = rigid_body_potential
        devlogger.info(f'Created ParticleType {name} @ {hex(id(self))}')
        
        for key in ParticleType.excludedAttributes:
            assert( key not in kwargs )

        for key,val in kwargs.items():
            self.__dict__[key] = val

    def is_same_type(self, other):
        assert(isinstance(other,ParticleType))
        if self == other:
            return True
        elif self.parent is not None and self.parent == other:
            return True
        else:
            return False

    def add_grid_potential(self, gridfile, scale=1):
        self.grid_potentials = getattr(self, 'grid_potentials', []) + [(gridfile,scale)]
        
    def __getattr__(self, name):
        """
        Try to get attribute from the parent

        """
        if "parent" not in self.__dict__ or self.__dict__["parent"] is None or name == "children":
           raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

        excluded_attributes = ParticleType.excludedAttributes
        if name in excluded_attributes:
            raise AttributeError("'{}' object has no attribute '{}' and cannot look it up from the parent".format(type(self).__name__, name))

        ## TODO: determine if there is a way to avoid __getattr__ if a method is being looked up
        try:
            ret = getattr(self.parent,name)
        except:
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))
        if ismethod(ret):
            raise AttributeError("'{}' object has no method '{}'".format(type(self).__name__, name))
        return ret 

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self
        
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

    def __eq__(a,b, check_equal = True):
        if check_equal: a._equal_check(b)
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
        self.rigid = False

        for key,val in kwargs.items():
            self.__dict__[key] = val
        
    def add_restraint(self, restraint):
        ## TODO: how to handle duplicating and cloning bonds
        self.restraints.append( restraint )

    def add_grid_potential(self, gridfile, scale=1):
        t0 = self.type_
        name = f'{t0.name}_g_{gridfile.replace(".dx","")}_s_{scale}'
        if t0.parent is not None:
            t = copy(t0)
            t.name = name
        else:
            t = ParticleType(name, parent=t0)
        t.add_grid_potential(gridfile, scale=scale)
        self.type_ = t
        self._clear_types()
        
    def get_restraints(self):
        return [(self,r) for r in self.restraints]

    def duplicate(self):
        new = deepcopy(self)
        return new

    def __getattr__(self, name):
        """
        First try to get attribute from the parent, then type_
        
        Note that this data structure seems to be fragile, can result in stack overflow
        
        """
        if name in ('__copy__','__deepcopy__'):
            ## Avoid using type_ and parent __copy__/__deepcopy__ functions!
            return None

        # return Child.__getattr__(self,name)
        try:
            return Child.__getattr__(self,name)
        except Exception as e:
            if 'type_' in self.__dict__:
                if name == 'parent':
                    raise Exception('Programming error')
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
            mass = p.mass
        except:
            mass = 1

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
                    type = p.type_.name[:7],
                    charge = p.charge,
                    mass = mass,
                    occupancy = occ,
                    beta = beta
                )
        return data

    def __copy__(self):
        Object.__copy__(self)

    def __repr__(self):
        return f'<{__name__}.{self.__class__.__name__} "{self.name}" of {self.type_}>'


class RigidBody(PointParticle):

    def __init__(self, type_, position, orientation, name="A", attached_particles=tuple(), **kwargs):
        parent = None
        if 'parent' in kwargs:
            parent = kwargs['parent']
        Child.__init__(self, parent=parent)
        Transformable.__init__(self,position, orientation)

        self.type_    = type_                
        self.idx     = None
        self.name = name
        self.counter = 0
        self.restraints = []
        self.attached_particles = []
        for p in attached_particles:
            self.attach_particle(p)
        
        for key,val in kwargs.items():
            self.__dict__[key] = val
        
    def add_restraint(self, restraint):
        raise NotImplementedError('Harmonic restraints are not yet supported for rigid bodies')
        ## TODO: how to handle duplicating and cloning bonds
        self.restraints.append( restraint )

    def get_restraints(self):
        return [(self,r) for r in self.restraints]

    def duplicate(self):
        new = deepcopy(self)
        return new

    def attach_particle(self, particle):
        """ The particle argument can be a PointParticle or Group (RigidBody children will be ignored). The position/orientation of the attached particle/group is in the RigidBody frame. """
        if particle.parent is not None:
            raise ValueError('RigidBody-attached particles are not allowed to have a parent')
        self.attached_particles.append( particle )

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
                raise AttributeError(r"'{type(self).__name__}' object has no attribute '{name}'")
    

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

    def write_pdb(self, filename, beta_from_fixed=False):
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

                pos = p.get_collapsed_position()
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

    def write_pqr(self, filename):
        if self.cacheInvalid:
            self._updateParticleOrder()
        with open(filename,'w') as fh:
            ## Write header
            fh.write("CRYST1{:>9.3f}{:>9.3f}{:>9.3f}  90.00  90.00  90.00 P 1           1\n".format( *self.dimensions ))

            ## Write coordinates
            formatString = "ATOM {idx:>6.6s} {name:^4.4s} {resname:3.3s} {chain:1.1s} {resid:>5.5s}   {x:.6f} {y:.6f} {z:.6f} {charge} {radius}\n"
            for p in self.particles:
                data = p._get_psfpdb_dictionary()

                idx = data['idx']
                if np.log10(idx) >= 5:
                    idx = " *****"
                else:
                    idx = "{:>6d}".format(idx)
                data['idx'] = idx

                x,y,z = p.get_collapsed_position()
                data['x'] = x
                data['y'] = y
                data['z'] = z
                assert(data['resid'] < 1e5)
                data['resid'] = "{:<4d}".format(data['resid'])
                if 'radius' not in data:
                    data['radius'] = 2 * (data['mass']/16)**0.333333
                fh.write( formatString.format(**data) )
        return
        
    def write_psf(self, filename):
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

    class _GroupSite():
        """ Class to represent a collection of particles that can be used by bond potentials """
        def __init__(self, particles, weights=None):
            if weights is not None:
                raise NotImplementedError
            self.particles = particles
            self.idx = None
            self.restraints = []
            
        def get_center(self):
            c = np.array((0,0,0))
            for p in self.particles:
                c = c + p.get_collapsed_position()
            c = c / len(self.particles)
            return c

        def add_restraint(self, restraint):
            self.restraints.append( restraint )
        def get_restraints(self):
            return [(self,r) for r in self.restraints]


    def __init__(self, children, origin=None, dimensions=(1000,1000,1000),
                 remove_duplicate_bonded_terms=True,
                 configuration=None, dummy_types=tuple(), **conf_params):

        PdbModel.__init__(self, children, dimensions, remove_duplicate_bonded_terms)
        self.origin = origin

        if configuration is None: 
            configuration = SimConf(**conf_params)
        self.configuration = configuration

        self.num_particles = 0
        self.particles = []
        self.type_counts = None

        self.dummy_types = dummy_types # TODO, determine whether these are really needed
        
        self.nonbonded_interactions = []
        self._nonbonded_interaction_files = [] # This could be made more robust

        self.cacheUpToDate = False

        self.group_sites = []

    def add_group_site(self, particles, weights=None):
        g = ArbdModel._GroupSite(particles, weights)
        self.group_sites.append(g)
        return g

    def _clear_types(self):
        devlogger.debug(f'{self}: Clearing types') 
        self.type_counts = None
        
    
    def clear_all(self, keep_children=False):
        Parent.clear_all(self, keep_children=keep_children)
        self.particles = []
        self.num_particles = 0
        self._clear_types()
        self.group_sites = []
        self._nonbonded_interaction_files = []

    def extend(self, other_model, copy=False):
        assert( isinstance(other_model, ArbdModel) )
        if copy == True:
            logger.warning(f'Forcing {self.__class__}.extend(other_model,copy=False)')
            copy = False

        ## Combine particle types, taking care to handle name clashes
        self._countParticleTypes()
        other_model._countParticleTypes()

        self.getParticleTypesAndCounts()
        other_model.getParticleTypesAndCounts()

        names = set([t.name for t in self.type_counts.keys()])

        devlogger.debug(f'Combining types {self.type_counts.keys()} and {other_model.type_counts.keys()}')
        for t1 in self.type_counts.keys():
            for t2 in other_model.type_counts.keys():
                if t1.name == t2.name and t1.__eq__(t2, check_equal=False):
                    i = 1
                    new_name = f'{t2.name}{i}'
                    while new_name in names:
                        i += 1
                        new_name = f'{t2.name}{i}'
                    devlogger.debug(f'Updating {t2.name} to {new_name}')
                    t2.name = new_name
                    names.add(new_name)
                    
        ## Combine interactions
        for i, tA, tB in other_model.nonbonded_interactions:
            devlogger.debug(f'Combining model interactions {i} {tA} {tB}')
            self.add_nonbonded_interaction(i,tA,tB)
                    
        # for g in other_model.children:
        #     self.update(g, copy=copy)
        g = Group()
        for attr in 'children position orientation bonds angles dihedrals impropers exclusions bond_angles product_potentials group_sites'.split():
            g.__setattr__(attr, other_model.__getattribute__(attr))
        devlogger.debug(f'Updating {self} with {g.children[0].children[0]}')
        self.update(g, copy=copy)

        self._clear_types()
        
        ## Combine configurations
        self.configuration = other_model.configuration.combine(self.configuration, policy='best', warn=True)
        
    def update(self, group , copy=False):
        assert( isinstance(group, Group) )
        if copy:
            group = deepcopy(group)
        group.parent = self
        self.add(group)
        
    def _get_nonbonded_interaction(self, typeA, typeB):
        for s,A,B in self.nonbonded_interactions:
            if A is None or B is None:
                if A is None and B is None:
                    return s
                elif A is None and typeB.is_same_type(B):
                    return s
                elif B is None and typeA.is_same_type(A):
                    return s
            elif typeA.is_same_type(A) and typeB.is_same_type(B):
                return s
        # raise Exception("No nonbonded scheme found for %s and %s" % (typeA.name, typeB.name))
        # print("WARNING: No nonbonded scheme found for %s and %s" % (typeA.name, typeB.name))

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
        for t in self.dummy_types:
            if t not in type_counts:
                type_counts[t] = 0

        for i,tA,tB in self.nonbonded_interactions:
            if tA is not None and tA not in type_counts:
                type_counts[tA] = 0
            if tB is not None and tB not in type_counts:
                type_counts[tB] = 0
            
        self.type_counts = type_counts
        devlogger.debug(f'{self}: Counting types: {type_counts}')
        
    def _updateParticleOrder(self):
        ## Create ordered list
        self.particles = [p for p in self if not p.rigid]
        # self.particles = sorted(particles, key=lambda p: (p.type_, p.idx))
        
        ## Update particle indices
        for i,p in enumerate(self.particles):
            p.idx = i

        ## TODO recurse through childrens' group_sites
        for i,g in enumerate(self.group_sites):
            g.idx = len(self.particles)+i
            
        # self.initialCoords = np.array([p.initialPosition for p in self.particles])

    def useNonbondedScheme(self, nbScheme, typeA=None, typeB=None):
        """ deprecated """
        logger.warning('useNonbondedScheme is deprecated! Please update your code to use `add_nonbonded_interaction`')
        self.add_nonbonded_interaction(nbScheme, typeA, typeB)

    def add_nonbonded_interaction(self, nonbonded_potential, typeA=None, typeB=None):
        self.nonbonded_interactions.append( [nonbonded_potential, typeA, typeB] )
        if typeA != typeB:
            self.nonbonded_interactions.append( [nonbonded_potential, typeB, typeA] )

    def prepare_for_simulation(self):
        ...
    
    def getParticleTypesAndCounts(self):
        ## TODO: remove(?)
        if self.type_counts is None:
            self._countParticleTypes()
            self._updateParticleOrder()

        return sorted( self.type_counts.items(), key=lambda x: x[0] )

    def _particleTypePairIter(self):
        typesAndCounts = self.getParticleTypesAndCounts()
        for i in range(len(typesAndCounts)):
            t1,n1 = typesAndCounts[i]
            if n1 == 0: continue
            for j in range(i,len(typesAndCounts)):
                t2,n2 = typesAndCounts[j]
                if n2 == 0: continue
                yield( [i,j,t1,t2] )

    def dimensions_from_structure( self, padding_factor=1.5, isotropic=False ):
        raise(NotImplementedError)

class SimConf():
    """ Class describing properties for a (ARBD or NAMD) simulation """

    def __init__(self, num_steps=None, output_period=None,
                 integrator=None, timestep=None, thermostat=None, barostat=None,
                 temperature=None, pressure=None,
                 cutoff=None, pairlist_distance=None, decomp_period=None, gpu=None,
                 seed=None, restart_file=None):

        self.num_steps = num_steps
        self.output_period = output_period

        self.integrator = integrator
        self.timestep = timestep
        self.thermostat = thermostat
        self.barostat = barostat

        self.temperature = temperature
        self.pressure = pressure
        self.cutoff = cutoff
        self.pairlist_distance = pairlist_distance
        self.decomp_period = decomp_period
        self.seed = seed
        self.restart_file = restart_file

        self.gpu = gpu
        
        # num_steps=100000000, timestep=None, output_period=1e4
        ...

    @property
    def temperature(self):
        return self.__temperature
    @temperature.setter
    def temperature(self,value):
        if value is not None and value <= 0:
            raise ValueError("Temperature must be positive")
        self.__temperature = value

    def combine(self, other, policy = 'override', warn=False):
        """ 
        Creates a new SimConf object whose properties are
        initialized to be from "self", but are overridden with
        properties in "other", provided they are not None
        """

        new_conf = copy(self)
        for attr in _get_properties_and_dict_keys(other):
            oldval = None
            val = other.__getattribute__(attr)
            if val is not None:
                try:
                    oldval = self.__getattribute__(attr)
                except:
                    pass
                if oldval != val and (oldval is not None) and \
                   (val is not None) and policy != 'override':
                    if policy == 'best':
                        if attr in ('timestep','output_period','decomp_period'):
                            if warn: logger.warning(f'Combining attribute {attr}: {oldval} != {val}, using {min([oldval,val])}')
                            new_conf.__setattr__(attr, min([oldval,val]))
                        elif attr in ('num_steps','cutoff','pairlist_distance'):
                            if warn: logger.warning(f'Combining attribute {attr}: {oldval} != {val}, using {max([oldval,val])}')
                            new_conf.__setattr__(attr, max([oldval,val]))
                        elif attr == 'integrator':
                            if 'MD' in (oldval,val) and 'BD' in (oldval,val):
                                if warn: logger.warning(f'Combining attribute {attr}: {oldval} != {val}, using "MD"')
                                new_conf.__setattr__(attr,'MD')
                            else:
                                logger.warning(f'Unsure how to combine {oldval} and {val} for {attr} under policy {policy}; using {val}')
                                new_conf.__setattr__(attr, val)
                        else:
                            logger.warning(f'Unsure how to combine {oldval} and {val} for {attr} under policy {policy}; using {val}')
                            new_conf.__setattr__(attr, val)                            
                    else:
                        raise ValueError(f'Unrecognized policy "{policy}" for combining SimConfs')
                else:
                    new_conf.__setattr__(attr, val)
        return new_conf

    def items(self):
        for attr in _get_properties_and_dict_keys(self):
            val = self.__getattribute__(attr)
            yield attr,val

class DefaultSimConf(SimConf):
    """ Generic class describing properties for a simulation """

    def __init__(self, num_steps=1e5, output_period=1e3,
                 integrator='MD', timestep=20e-6, thermostat='Langevin', barostat=None,
                 temperature=295, pressure=1,
                 cutoff=50, pairlist_distance=None, decomp_period=40,
                 seed=None, restart_file=None, gpu=0):
        SimConf.__init__(self, num_steps=num_steps, output_period=output_period,
                         integrator=integrator, timestep=timestep, thermostat=thermostat, barostat=barostat,
                         temperature=temperature, pressure=pressure,
                         cutoff=cutoff, pairlist_distance=pairlist_distance, decomp_period=decomp_period,
                         seed=seed, restart_file=restart_file, gpu=gpu)
        
        self.num_steps = num_steps
        self.output_period = output_period
        self.__temperature = temperature
        self.pressure = pressure
        
        # num_steps=100000000, timestep=None, output_period=1e4
        ...

    @property
    def temperature(self):
        return self.__temperature
    @temperature.setter
    def temperature(self,value):
        if (value <= 0):
            raise ValueError("Temperature must be positive")
        self.__temperature = value
        

class SimEngine(metaclass=ABCMeta):
    """ Abstract class for running a simulation of a model """
    def __init__(self, configuration=None):
        self.configuration = configuration

    @property
    @abstractmethod
    def default_binary(self):
        ...

    @abstractmethod
    def _generate_command_string(self, binary, output_name, output_directory, gpu=0, replicas=1):
        ...

    @abstractmethod
    def write_simulation_files(self, model, output_name):
        ...

    @abstractmethod
    def get_default_conf(self):
        ...
        
    def simulate(self, model, output_name, output_directory='output',
                 directory='.', log_file=None,
                 binary=None, num_procs=None, dry_run = False, configuration = None, replicas = 1, **conf_params):

        ## TODO: Allow _get_combined_conf to take certain parameters as arguments, or otherwise refactor to make this more elegant
        gpu = self._get_combined_conf(model, **conf_params).gpu
        assert(type(gpu) is int)

        if num_procs is None:
            import multiprocessing
            num_procs = max(1,multiprocessing.cpu_count()-1)

        d_orig = os.getcwd()
        try:
            model._d_orig = d_orig
            if not os.path.exists(directory):
                os.makedirs(directory)
            os.chdir(directory)

            model.prepare_for_simulation()

            if output_directory == '': output_directory='.'

            if dry_run:
                if binary is None: binary=self.default_binary
            else:
                binary = self._get_binary(binary)

            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            elif not os.path.isdir(output_directory):
                raise Exception("output_directory '%s' is not a directory!" % output_directory)

            self.write_simulation_files(model, output_name, configuration, **conf_params)

            ## http://stackoverflow.com/questions/18421757/live-output-from-subprocess-command
            cmd = self._generate_command_string(binary, output_name, output_directory, num_procs, gpu, replicas)

            if dry_run:
                logger.info(f'Run with: {" ".join(cmd)}')
            else:
                logger.info(f'Running {self.default_binary} with: {" ".join(cmd)}')
                if log_file is None or (hasattr(log_file,'write') and callable(log_file.write)):
                    fd = sys.stdout if log_file is None else log_file
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=False) # , universal_newlines=True)
                    ## re-open to get \r recognized as new line
                    with open(os.dup(process.stdout.fileno()), newline='') as nice_stdout:
                        for line in nice_stdout:
                            try:
                                fd.write(line)
                            except:
                                print("WARNING: could not encode line; your locale might not be set correctly")
                            fd.flush()
                else:
                    with open(log_file,'w') as fd:
                        process = subprocess.Popen(cmd, stdout=log_file, universal_newlines=True)
                        process.communicate()

        except:
            raise
        finally:
            del model._d_orig
            os.chdir(d_orig)

    def _get_binary(self, binary=None):
        if binary is None:
            for path in os.environ["PATH"].split(os.pathsep):
                path = path.strip('"')
                fname = os.path.join(path, self.default_binary)
                if os.path.isfile(fname) and os.access(fname, os.X_OK):
                    binary = fname
                    break

        if binary is None: raise Exception("{} was not found".format(self.default_binary))

        if not os.path.exists(binary):
            raise Exception("{} was not found".format(self.default_binary))
        if not os.path.isfile(binary):
            raise Exception("{} was not found".format(self.default_binary))
        if not os.access(binary, os.X_OK):
            raise Exception("{} is not executable".format(self.default_binary))

        return binary

    def _get_combined_conf(self, model, **conf_params):
        conf = self.get_default_conf() 
        conf = conf.combine(model.configuration)
        conf = conf.combine(self.configuration)
        conf = conf.combine(SimConf(**conf_params))
        return conf

    """ TODO: Remove
    # def _configuration_as_dict(self, model):
    #     params = dict()
    #     for o in (self.configuration, self, model):
    #         for k in _get_properties_and_dict_keys(o):
    #             params[k] = o.__get_attribute__(k)
    #     return params
    """

class ArbdEngine(SimEngine):
    """ Interface to ARBD simulation engine """
    def __init__(self, extra_bd_file_lines="", configuration=None, **conf_params):
        
        self.extra_bd_file_lines = extra_bd_file_lines
        
        if configuration is None: 
            configuration = SimConf(**conf_params)
        SimEngine.__init__(self, configuration)

        self.num_particles = 0
        self.particles = []     # TODO decide if this should belong here, or with model
        self.type_counts = None # TODO decide if this should belong here, or with model

        self._written_bond_files = dict()        

        self.cacheUpToDate = False

    @property
    def default_binary(self):
        return 'arbd'

    def _generate_command_string(self, binary, output_name, output_directory, num_procs=None, gpu=0, replicas=1):
        cmd = [binary, '-g', "%d" % gpu]
        if replicas > 1:
            cmd = cmd + ['-r',replicas]
        cmd = cmd + ["%s.bd" % output_name, "%s/%s" % (output_directory, output_name)]
        cmd = tuple(str(x) for x in cmd)
        return cmd

    def get_default_conf(self):
        conf = SimConf(num_steps=1e5, output_period=1e3,
                 integrator='MD', timestep=20e-6, thermostat='Langevin', barostat=None,
                 temperature=295, pressure=1,
                 cutoff=50, pairlist_distance=None, decomp_period=40, gpu = 0,
                 seed=None, restart_file=None)
        return conf
    
    # -------------------------- #
    # Methods for printing model #
    # -------------------------- #

    def write_simulation_files(self, model, output_name, configuration=None, **conf_params):
        
        if configuration is None:
            configuration = self._get_combined_conf(model, **conf_params)
        
        ## TODO: save and reference directories and prefixes using member data
        d = self.potential_directory = "potentials"
        if not os.path.exists(d):
            os.makedirs(d)
 
        model.write_psf( output_name+'.psf' )
        model.write_pdb( output_name+'.pdb' )

        self._write_particle_file( model, output_name + ".particles.txt", configuration )

        self._write_restraint_file( model, "{}/{}.restraint.txt".format(d, output_name) )
        self._write_bond_file( model, "{}/{}.bond.txt".format(d, output_name) )
        self._write_angle_file( model, "{}/{}.angle.txt".format(d, output_name) )
        self._write_dihedral_file( model, "{}/{}.dihedral.txt".format(d, output_name) )
        self._write_exclusion_file( model, "{}/{}.exclusion.txt".format(d, output_name) )
        self._write_bond_angle_file( model, f"{d}/{output_name}.bond-angle.txt" )
        self._write_product_potential_file( model, f"{d}/{output_name}.product_potential.txt" )
        self._write_group_sites_file( model, f"{d}/{output_name}.group_sites.txt" )

        self._write_potential_files( model, output_name, directory = d, configuration = configuration )
        self._write_conf( model, output_name, configuration, directory = d )
        ## , numSteps=numSteps, outputPeriod=outputPeriod, restart_file=restart_file )

    def _write_particle_file(self, model, filename, configuration=None, **conf_params):
        if configuration is None:
            configuration = self._get_combined_conf(model, **conf_params)

        with open(filename,'w') as fh:
            if configuration.integrator in ('Brown', 'Brownian', 'BD'):
                for p in model.particles:
                    data = tuple([p.idx,p.type_.name] + [x for x in p.get_collapsed_position()])
                    fh.write("ATOM %d %s %f %f %f\n" % data)
            else:
                for p in model.particles:
                    data = [p.idx,p.type_.name] + [x for x in p.get_collapsed_position()]
                    try:
                        data = data + p.momentum
                    except:
                        try:
                            data = data + p.velocity*p.mass
                        except:
                            data = data + [0,0,0]
                    fh.write("ATOM %d %s %f %f %f %f %f %f\n" % tuple(data))
                
    def _write_rigid_group_file(self, model, filename, groups):
        with open(filename,'w') as fh:
            for g in groups:
                fh.write("#Group\n")
                try:
                    if len(g.trans_damping) != 3: raise
                    fh.write(" ".join(str(v) for v in g.trans_damping) + " ")
                except:
                    raise ValueError("Group {} lacks 3-value 'trans_damping' attribute")
                try:
                    if len(g.rot_damping) != 3: raise
                    fh.write(" ".join(str(v) for v in g.rot_damping) + " ")
                except:
                    raise ValueError("Group {} lacks 3-value 'rot_damping' attribute")
                fh.write("{}\n".format(len(g)))
                particles = [p for p in g]

                def chunks(l, n):
                    """Yield successive n-sized chunks from l."""
                    for i in range(0, len(l), n):
                        yield l[i:i + n]

                for c in chunks(particles,8):
                    fh.write(" ".join(str(p.idx) for p in c) + "\n")


    def getParticleTypesAndCounts(self):
        ## TODO: remove()?
        return sorted( model.type_counts.items(), key=lambda x: x[0] )

    def _particleTypePairIter(self):
        typesAndCounts = self.getParticleTypesAndCounts()
        for i in range(len(typesAndCounts)):
            t1 = typesAndCounts[i][0]
            for j in range(i,len(typesAndCounts)):
                t2 = typesAndCounts[j][0]
                yield( (i,j,t1,t2) )
    
    def _write_potential_files(self, model, prefix, directory = "potentials", configuration=None, **conf_params):
        try: 
            os.makedirs(directory)
        except OSError:
            if not os.path.isdir(directory):
                raise

        if configuration is None:
            configuration = self._get_combined_conf(model, **conf_params)

        path_prefix = "{}/{}".format(directory,prefix)
        self._write_nonbonded_parameter_files( model, path_prefix + "-nb", configuration )
                
    def _write_nonbonded_parameter_files(self, model, prefix, configuration=None, **conf_params):
        if configuration is None:
            configuration = self._get_combined_conf(model, **conf_params)

        model._nonbonded_interaction_files = [] # clear old nb files

        x = np.arange(0, configuration.cutoff)
        for i,j,t1,t2 in model._particleTypePairIter():
            f = "%s.%s-%s.dat" % (prefix, t1.name, t2.name)
            interaction = model._get_nonbonded_interaction(t1,t2)
            devlogger.debug(f'_write_nonbonded_parameter_files: {i}, {j}, {t1}, {t2}, {interaction}')
            if interaction is None:
                model._nonbonded_interaction_files.append(None)
                continue
            old_range = interaction.range_

            interaction.range_ = [0, configuration.cutoff]
            interaction.write_file(f, (t1, t2))
            interaction.range_ = old_range 

            model._nonbonded_interaction_files.append(f)
        devlogger.debug(f'model._nonbonded_interaction_files: {model._nonbonded_interaction_files}')

    def _write_restraint_file( self, model, filename ):
        self._restraint_filename = filename
        with open(self._restraint_filename,'w') as fh:
            for i,restraint in model.get_restraints():
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

    def _write_bond_file( self, model, filename ):
        self._bond_filename = filename
        for b in list( set( [b for i,j,b,ex in model.get_bonds()] ) ):
            if type(b) is not str and not isinstance(b, Path):
                b.write_file()

        with open(self._bond_filename,'w') as fh:
            for i,j,b,ex in model.get_bonds():
                try:
                    bfile = b.filename()
                except:
                    bfile = str(b)
                item = (i.idx, j.idx, bfile)
                if ex:
                    fh.write("BOND REPLACE %d %d %s\n" % item)
                else:
                    fh.write("BOND ADD %d %d %s\n" % item)

    def _write_angle_file( self, model, filename ):
        self._angle_filename = filename
        for b in list( set( [b for i,j,k,b in model.get_angles()] ) ):
            if type(b) is not str and not isinstance(b, Path):
                b.write_file()

        with open(self._angle_filename,'w') as fh:
            for b in model.get_angles():
                try:
                    bfile = b[-1].filename()
                except:
                    bfile = str(b[-1])
                item = tuple([p.idx for p in b[:-1]] + [bfile])
                fh.write("ANGLE %d %d %d %s\n" % item)

    def _write_dihedral_file( self, model, filename ):
        self._dihedral_filename = filename
        for b in list( set( [b for i,j,k,l,b in model.get_dihedrals()] ) ):
            if type(b) is not str and not isinstance(b, Path):
                b.write_file()

        with open(self._dihedral_filename,'w') as fh:
            for b in model.get_dihedrals():
                try:
                    bfile = b[-1].filename()
                except:
                    bfile = str(b[-1])
                item = tuple([p.idx for p in b[:-1]] + [bfile])
                fh.write("DIHEDRAL %d %d %d %d %s\n" % item)

    def _write_exclusion_file( self, model, filename ):
        self._exclusion_filename = filename
        with open(self._exclusion_filename,'w') as fh:
            for ex in model.get_exclusions():
                item = tuple(int(p.idx) for p in ex)
                fh.write("EXCLUDE %d %d\n" % item)

    def _write_bond_angle_file( self, model, filename ):
        self._bond_angle_filename = filename
        if len(model.bond_angles) > 0:
            with open(self._bond_angle_filename,'w') as fh:
                for b in model.get_bond_angles():
                    bfiles = []
                    for p in b[-1]:
                        try:
                            bfile = p.filename()
                        except:
                            bfile = str(p)
                        bfiles.append(bfile)
                    item = tuple([p.idx for p in b[:-1]] + bfiles)
                    fh.write("BONDANGLE %d %d %d %d %s %s %s\n" % item)

    def _write_product_potential_file( self, model, filename ):
        self._product_potential_filename = filename
        if len(model.product_potentials) > 0:
            with open(self._product_potential_filename,'w') as fh:
                for pot in model.get_product_potentials():
                    line = "PRODUCTPOTENTIAL "
                    for ijk_tb in pot:
                        ijk = ijk_tb[:-1]
                        tb = ijk_tb[-1]
                        if type(tb) is tuple or type(tb) is list:
                            if len(tb) != 2: raise ValueError("Invalid product potential")
                            type_,b = tb
                            if type(type_) is not str: raise ValueError("Invalid product potential: unrecognized specification of potential type")
                        else:
                            type_ = ""
                            b = tb
                        if type(b) is not str and not isinstance(b, Path):
                            b.write_file()
                        try:
                            bfile = b.filename()
                        except:
                            bfile = str(b)
                        line = line+" ".join([str(x.idx) for x in ijk])+" "
                        line = line+" ".join([str(x) for x in [type_,bfile] if x != ""])+" "
                    fh.write(line)

    def _write_group_sites_file( self, model, filename ):
        self._group_sites_filename = filename
        if len(model.group_sites) > 0:
            with open(self._group_sites_filename,'w') as fh:
                for i,g in enumerate(model.group_sites):
                    assert( i+len(model.particles) == g.idx )
                    ids = " ".join([str(int(p.idx)) for p in g.particles])
                    fh.write("GROUP %s\n" % ids)

    def _write_conf(self, model, prefix, configuration=None, **conf_params):
        # num_steps=100000000, output_period=10000, restart_file=None,
        ## TODO: raise exception if _writeArbdPotentialFiles has not been called
        filename = f'{prefix}.bd'

        ## Prepare a dictionary to fill in placeholders in the configuration file
        if configuration is None:
            configuration = self._get_combined_conf(model, **conf_params)

        params = dict()
        for k,v in configuration.items():
            params[k] = v

        if configuration.seed is None:
            params['seed']     = f'seed {int(np.random.default_rng().integers(1,99999,1))}'
        else:
            params['seed'] = "seed {:d}".format(configuration.seed)
        params['num_steps']       = int(configuration.num_steps)

        # params['coordinateFile'] = "%s.coord.txt" % prefix
        params['particle_file'] = "%s.particles.txt" % prefix
        if configuration.restart_file is None:
            params['restart_coordinates'] = ""
        else:
            params['restart_coordinates'] = "restart_coordinates %s" % configuration.restart_file

        for k,v in zip('XYZ', model.dimensions):
            params['dim'+k] = v

        if model.origin is None:
            for k,v in zip('XYZ', model.dimensions):
                params['origin'+k] = -v*0.5
        else:
            for k,v in zip('XYZ', model.origin):
                params['origin'+k] = v
             
        if params['pairlist_distance'] is None:
            params['pairlist_distance'] = 10
        else:
            params['pairlist_distance'] -= params['cutoff'] 

        """ Find rigid groups """
        rigid_groups = []
        def get_rigid_groups(parent):
            ret_list = []
            for c in parent.children:
                is_rigid = c.rigid if 'rigid' in c.__dict__ else False
                if is_rigid:
                    rigid_groups.append(c)
                elif isinstance(c,Group):
                    get_rigid_groups(c)
        get_rigid_groups(model)

        if len(rigid_groups) > 0:
            rb_group_filename = "{}.rb-group.txt".format(prefix)
            params['integrator'] = """FusDynamics
groupFileName {}
scaleFactor 0.05""".format(rb_group_filename)
            self._write_rigid_group_file(rb_group_filename, rigid_groups)

        if params['integrator'] == 'MD':
            params['integrator'] = 'Langevin'

        ## Actually write the file
        with open(filename,'w') as fh:
            fh.write("""{seed}
timestep {timestep}
steps {num_steps}
numberFluct 0                   # deprecated

interparticleForce 1            # other values deprecated
fullLongRange 0                 # deprecated
temperature {temperature}
ParticleDynamicType {integrator}

outputPeriod {output_period}
## Energy doesn't actually get printed!
outputEnergyPeriod {output_period}
outputFormat dcd

## Infrequent domain decomposition because this kernel is still very slow
decompPeriod {decomp_period}
cutoff {cutoff}
pairlistDistance {pairlist_distance}

origin {originX} {originY} {originZ}
systemSize {dimX} {dimY} {dimZ}

{extra_bd_file_lines}
\n""".format(extra_bd_file_lines=self.extra_bd_file_lines, **params))
            
            ## Write entries for each type of particle
            for pt,num in model.getParticleTypesAndCounts():
                if num == 0: continue
                devlogger.debug(f'Writing configuration for particle type {pt}')
                ## TODO create new particle types if existing has grid
                particleParams = pt.__dict__.copy()
                particleParams['num'] = num
                if configuration.integrator in ('Brown', 'Brownian', 'BD'):
                    try:
                        D = pt.diffusivity
                    except:
                        """ units "k K/(amu/ns)" "AA**2/ns" """
                        D = 831447.2 * configuration.temperature / (pt.mass * pt.damping_coefficient)
                    particleParams['dynamics'] = 'diffusion {D}'.format(D = D)
                elif configuration.integrator in ('MD','Langevin','FusDynamic'):
                    try:
                        gamma = pt.damping_coefficient
                    except:
                        """ units "k K/(AA**2/ns)" "amu/ns" """
                        gamma = 831447.2 * configuration.temperature / (pt.mass*pt.diffusivity)
                    particleParams['dynamics'] = """mass {mass}
transDamping {g} {g} {g}
""".format(mass=pt.mass, g=gamma)
                else:
                    raise ValueError("Unrecognized particle integrator '{}'".format(configuration.particle_integrator))
                if pt.rigid_body_potential is not None:
                    particleParams['rigid_potential'] = "rigidBodyPotential {}".format(pt.rigid_body_potential)
                else:
                    particleParams['rigid_potential'] = ''
                fh.write("""
particle {name}
num {num}
{dynamics}
{rigid_potential}
""".format(**particleParams))
                if 'grid_potentials' in particleParams:
                    grids = []
                    scales = []
                    for g,s in pt.grid_potentials:
                        tmp_g = str(g) if str(g)[0] == '/' else Path(model._d_orig) / g
                        try:
                            grids.append( os.path.relpath(str(tmp_g)) )
                        except:
                            devlogger.info(f'Relative path for {pt} grid not found... using {tmp_g}')
                            grids.append(str(tmp_g))
                            
                        scales.append(str(s))

                    fh.write("gridFile {}\n".format(" ".join(grids)))
                    fh.write("gridFileScale {}\n".format(" ".join(scales)))
                else:
                    fh.write("gridFile {}/null.dx\n".format(self.potential_directory))
                if 'rb_grid' in particleParams:
                    grids = []
                    scales = []
                    for entry in pt.rb_grid:
                        try:
                            g, s = entry
                        except:
                            g = entry
                            s = 1
                        ## TODO, use Path.relative_to?
                        try:
                            grids.append(str( g.relative_to(os.getcwd()) ))
                        except:
                            grids.append(str(g))
                        scales.append(s)
                    fh.write("rigidBodyPotential {}\n".format(" ".join(grids)))
                    if any([s!=1 for s in scales]):
                        raise NotImplementedError

            ## Write coordinates and interactions
            fh.write("""
## Input coordinates
inputParticles {particle_file}
{restart_coordinates}

## Interaction potentials
tabulatedPotential  1
## The i@j@file syntax means particle type i will have NB interactions with particle type j using the potential in file
""".format(**params))
            for pair,f in zip(model._particleTypePairIter(), model._nonbonded_interaction_files):
                if f is not None:
                    i,j,t1,t2 = pair
                    fh.write("tabulatedFile %d@%d@%s\n" % (i,j,f))

            ## Bonded interactions
            restraints = model.get_restraints()
            bonds = model.get_bonds()
            angles = model.get_angles()
            dihedrals = model.get_dihedrals()
            exclusions = model.get_exclusions()
            bond_angles = model.get_bond_angles()
            prod_pots = model.get_product_potentials()
            # group_sites = model.get_group_sites()
            group_sites = model.group_sites

            if len(bonds) > 0:
                for b in model._get_bond_potentials():
                    try:
                        bfile = b.filename()
                    except:
                        bfile = str(b)
                    fh.write("tabulatedBondFile %s\n" % bfile)

            if len(angles) > 0:
                for b in model._get_angle_potentials():
                    try:
                        bfile = b.filename()
                    except:
                        bfile = str(b)
                    fh.write("tabulatedAngleFile %s\n" % bfile)

            if len(dihedrals) > 0:
                for b in list(set([b for i,j,k,l,b in dihedrals])):
                    try:
                        bfile = b.filename()
                    except:
                        bfile = str(b)
                    fh.write("tabulatedDihedralFile %s\n" % bfile)

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
            if len(bond_angles) > 0:
                fh.write("inputBondAngles %s\n" % self._bond_angle_filename)
            if len(prod_pots) > 0:
                fh.write("inputProductPotentials %s\n" % self._product_potential_filename)
            if len(group_sites) > 0:
                fh.write("inputGroups %s\n" % self._group_sites_filename)
     
        write_null_dx = False
        for pt,num in model.getParticleTypesAndCounts():
            if num == 0: continue
            if "grid_potentials" not in pt.__dict__:
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

class NamdEngine(SimEngine):
    """ Partial interface to NAMD simulation engine """

    def __init__(self, configuration=None, **conf_params):
        if configuration is None: 
            configuration = SimConf(**conf_params)
        SimEngine.__init__(self, configuration)

        self.num_particles = 0
        self.particles = []     # TODO decide if this should belong here, or with model
        self.type_counts = None # TODO decide if this should belong here, or with model
        self.cacheUpToDate = False

    @property
    def default_binary(self):
        return 'namd2'

    def _generate_command_string(self, binary, output_name, output_directory, num_procs=1, gpu=None, replicas=1):
        cmd = [binary]
        if gpu is not None and len(gpu) > 0:
            cmd.extend(['+devices'] + [','.join([str(int(g)) for g in gpu])])
        cmd.extend([f'+p{str(num_procs)}'])    
        if replicas > 1:
            raise NotImplementedError
        cmd = cmd + [f'{output_name}.namd']
        cmd = tuple(str(x) for x in cmd)
        return cmd

    def get_default_conf(self):
        conf = SimConf(num_steps=1e5, output_period=1e3,
                 integrator='MD', timestep=2e-6, thermostat='Langevin', barostat=None,
                 temperature=295, pressure=1,
                 cutoff=12, pairlist_distance=14, decomp_period=12,
                 seed=None, restart_file=None)
        return conf

    def write_simulation_files(self, model, output_name, configuration=None, write_pqr=False, copy_ff_from=get_resource_path("charmm36.nbfix"), **conf_params):
        if configuration is None:
            configuration = self._get_combined_conf(model, **conf_params)

            model.write_psf( output_name+'.psf' )
            model.write_pdb( output_name+'.pdb' )
            model.write_pdb( output_name + ".fixed.pdb", beta_from_fixed=True )
            if write_pqr: model.write_pqr( output_name + ".pqr" )        

            if copy_ff_from is not None and copy_ff_from != '':
                try:
                    shutil.copytree( copy_ff_from, Path(copy_ff_from).stem )
                except FileExistsError:
                    pass

            self._write_conf( model, output_name, configuration )

    def write_conf( self, output_name, minimization_steps=4800, num_steps = 1e6,
                    output_directory = 'output',
                    update_dimensions=True, extrabonds=True ): 

        """ Write a NAMD configuration file (developed for the mrdna package) """
        
        num_steps = int(num_steps//12)*12
        minimization_steps = int(minimization_steps//24)*24
        if num_steps < 12:
            raise ValueError("Must run with at least 12  steps")
        if minimization_steps < 24:
            raise ValueError("Must run with at least 24 minimization steps")

        format_data = model.__dict__.copy() # get parameters from System object

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
            format_data['dimensions'] = model.dimensions_from_structure()

        for k,v in zip('XYZ', format_data['dimensions']):
            format_data['origin'+k] = -v*0.5
            format_data['cell'+k] = v

        format_data['prefix'] = output_name
        format_data['minimization_steps'] = int(minimization_steps//2)
        format_data['num_steps'] = num_steps
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
    minimize {minimization_steps:d}
    fixedAtoms off
    minimize {minimization_steps:d}
}} else {{
    bincoordinates  {output_directory}/$prefix-$nLast.restart.coor
    binvelocities   {output_directory}/$prefix-$nLast.restart.vel
}}

run {num_steps:d}
""".format(**format_data))
