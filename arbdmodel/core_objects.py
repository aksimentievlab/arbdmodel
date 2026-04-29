# -*- coding: utf-8 -*-
import numpy as np
from inspect import ismethod
from copy import copy, deepcopy
from pathlib import Path
from .logger import logger, get_resource_path,devlogger
from .interactions import SwitchPotential

class Citation():
    def __init__(self, author=None, title=None, journal=None, volume=None, number=None, pages=None, year=None, doi=None):
        self.author = author
        self.title = title
        self.journal = journal
        self.volume = volume
        self.number = number
        self.pages = pages
        self.year = year
        self.doi = doi

    def display(self):
        msg = ''
        if self.author: msg = f'{self.author}. '
        if self.title: msg = f'{msg}{self.title}. '
        if self.journal: msg = f'{msg}{self.journal} '
        if self.volume:
            msg = f'{msg} {self.volume}{"," if any((self.number, self.pages, self.year)) else "."} '
            if self.number:
                msg = f'{msg}({self.number}){"" if any((self.pages,self.year)) else "."} '
            if self.pages: msg = f'{msg}{self.pages}{"" if self.year else "."} '
        if self.year: msg = f'{msg}({self.year}). '
        if self.doi:
            if self.doi[:4] == 'http':
                msg = f'{msg}{self.doi}'
            else:
                msg = f'{msg}https://doi.org/{self.doi}'
        return msg


## Abstract classes
class Transformable():
    """
    A class for objects that can be transformed in 3D space.
    The Transformable class provides basic functionality for positioning and orienting
    objects in 3D space, as well as methods for applying transformations such as
    translations and rotations.

    Parameters:
        position (numpy.ndarray): The position of the object in 3D space.
        orientation (numpy.ndarray or None): The orientation matrix of the object.
            If None, the object has no orientation.

    Note:
        - The class is designed to work with the Child class for hierarchical transformations.
        - Orientation is represented as a rotation matrix.
        - All transformation methods modify the object in place.
    """
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

class BondedTerm():
    def __init__(self, particles, potential, switch_schedule=None):
        if len(particles) != len(set(particles)):
            raise ValueError('BondedTerm particles included duplicates')

        self.particles = particles
        self.potential = potential

        if switch_schedule is None: switch_schedule = []
        if len(switch_schedule) > 0 and not isinstance(potential,SwitchPotential):
            raise ValueError("Bonded term's potential is not switchable")
        self.switch_schedule = switch_schedule

    def add_switch_step(self, step, value):
        if  not isinstance(self.potential,SwitchPotential):
            raise ValueError("Bonded term's potential is not switchable")

        if step < 0:
            raise ValueError(f'Potential switching step should be non-negative')

        if len(self.switch_schedule) > 0 and self.switch_schedule[-1][0] >= step:
            raise ValueError(f'Potential switching step mut occur later than previous step')

        if step != int(step):
            raise ValueError('Potential switching step must be an integer')

        if value != float(value) or value < 0 or value > 1:
            raise ValueError('Potential switching value must be number between zero and one')

        self.switch_schedule.append((step,value))

    _warn = True
    def __iter__(self):
        if self.__class__._warn:
            devlogger.warning('Expanding bonded term for backward compatibility will be deprecated')
            self.__class__._warn = False

        for p in self.particles: yield p
        yield self.potential
        try: yield self.axis
        except: pass
        try: yield self.exclusion_flag
        except: pass
        yield self.switch_schedule

    def __getitem__(self, index):
        return list(self)[index]


class BondTerm(BondedTerm):
    def __init__(self, p1, p2, potential, exclusion_flag=False, switch_schedule=None):
        BondedTerm.__init__(self, (p1,p2), potential, switch_schedule)
        self.exclusion_flag = exclusion_flag
class AngleTerm(BondedTerm):
    def __init__(self, p1, p2, p3, potential, switch_schedule=None):
        BondedTerm.__init__(self, (p1,p2,p3), potential, switch_schedule)
class DihedralTerm(BondedTerm):
    def __init__(self, p1, p2, p3, p4, potential, switch_schedule=None):
        BondedTerm.__init__(self, (p1,p2,p3,p4), potential, switch_schedule)
class BondXYTerm(BondTerm):
    def __init__(self, p1, p2, potential, axis, exclusion_flag=False, switch_schedule=None):
        BondTerm.__init__(self, (p1,p2), potential, exclusion_flag, switch_schedule)
        try:    hash(axis)
        except: axis = tuple(axis)
        self.axis = axis

class NonbondedTermList(list):
    def append(self, item):
        if isinstance(item,NonbondedTerm):
            list.append(self,item)
        else:
            list.append(self,NonbondedTerm(*item))

class NonbondedTerm():
    def __init__(self, potential, type1=None, type2=None):
        self.potential = potential
        self.type1 = type1
        self.type2 = type2

    _warn = True
    def __iter__(self):
        if self.__class__._warn:
            devlogger.warning('Expanding nonbonded term for backward compatibility will be deprecated')
            self.__class__._warn = False

        yield self.potential
        yield self.type1
        yield self.type2

    def __getitem__(self, index):
        return list(self)[index]


class Parent():
    """
    The Parent class implements a hierarchical tree structure for organizing objects in a simulation.
    This class serves as a container for child objects and provides methods for managing relationships, properties, and interactions between these objects within a molecular simulation context.

    Parameters:
        children (list): A list of child objects belonging to this parent.
        remove_duplicate_bonded_terms (bool): Whether to remove duplicate bonded terms when collecting them.
        bonds (list): List of bond interactions between particles.
        angles (list): List of angle interactions between triplets of particles.
        dihedrals (list): List of dihedral interactions between quartets of particles.
        vector_angles (list): List of vector angle interactions.
        impropers (list): List of improper dihedral interactions.
        exclusions (list): List of pairwise exclusions from non-bonded interactions.
        bond_angles (list): List of combined bond-angle interactions.
        product_potentials (list): List of product potentials.
        group_sites (list): List of group sites.
    """
    def __init__(self, children=None, remove_duplicate_bonded_terms=False):
        self.children = []
        if children is not None:
            if not any(isinstance(children, _cls) for _cls in (tuple,list)):
                raise ValueError(f'Arddmodel.Parent(children,...): children is not a list or tuple; did you pass a list?')
            for x in children:
                self.add(x)
        
        self.remove_duplicate_bonded_terms = remove_duplicate_bonded_terms
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.bondXY = []
        self.vector_angles = []
        self.impropers = []
        self.exclusions = []
        self.vector_angles = []
        self.bond_angles = []
        self.product_potentials = []
        self.group_sites = []
        
        ## TODO: self.cacheInvalid = True # What will be in the cache?

    def add(self,x):
        """
        Adds a Child object to this group.
        
        This method establishes a parent-child relationship between the current object
        and the provided Child object. It checks that the provided object is a Child,
        and that it doesn't already belong to another parent.
        
        Parameters
        ----------
        x : Child
            The Child object to add to this group
            
        Raises
        ------
        Exception
            If x is not an instance of Child
            If x already belongs to another parent
        """

        if not isinstance(x,Child):
            raise Exception('Attempted to add an object to a group that does not inherit from the "Child" type')

        if x.parent is not None and x.parent is not self:
            raise Exception("Child {} already belongs to some group".format(x))
        x.parent = self
        self.children.append(x)

    def insert(self,idx,x):
        """
        Insert a child object at a specific position in the group's children list.

        This method inserts a new child object into the group at the specified index.
        It performs checks to ensure the object is a valid Child type and that it
        doesn't already belong to another parent group to maintain the integrity
        of the parent-child hierarchy.

        Parameters
        ----------
        idx : int
            The index position where the child should be inserted.
        x : Child
            The child object to be inserted. Must inherit from the Child class.

        Raises
        ------
        Exception
            If the provided object doesn't inherit from the Child class or
            if the child already belongs to another parent group.
        """
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
        self.bondXY = []
        self.vector_angles = []
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

    def add_bond(self, i,j, bond, exclude=False, switch_schedule=None):
        # beads = [b for b in self]
        # for b in (i,j): assert(b in beads)
        self.bonds.append( BondTerm(i,j, bond, exclude, switch_schedule) )

    def add_angle(self, i,j,k, angle, switch_schedule=None):
        self.angles.append( AngleTerm(i,j,k, angle, switch_schedule) )

    def add_dihedral(self, i,j,k,l, dihedral, switch_schedule=None):
        self.dihedrals.append( DihedralTerm(i,j,k,l, dihedral, switch_schedule) )

    def add_bondXY(self, i,j, bond, axis, exclude=False, switch_schedule=None):
        assert( i is not j )
        try:    hash(axis)
        except: axis = tuple(axis)
        self.bondXY.append( BondXYTerm(i,j, bond, axis, exclude, switch_schedule) )

    def add_vector_angle(self, i,j,k,l, potential, switch_schedule=None):
        self.vector_angles.append( DihedralTerm(i,j,k,l, potential, switch_schedule) )

    def add_improper(self, i,j,k,l, dihedral, switch_schedule=None):
        self.impropers.append( DihedralTerm(i,j,k,l, dihedral, switch_schedule) )

    def add_exclusion(self, i,j):
        ## TODO: how to handle duplicating and cloning bonds
        ## TODO: perform following check elsewhere
        # beads = [b for b in self]
        # for b in (i,j): assert(b in beads)
        self.exclusions.append( (i,j) )

    def add_bond_angle(self, i,j,k,l, bond_angle, exclude=False, switch_schedule=None):
        self.bond_angles.append( BondAngle(i,j,k,l, bond_angle, exclude, switch_schedule) )

    def add_product_potential(self, potential_list):
        """ potential_list: list or tuple of BondedTerm objects; only basic bonded terms are supported  """
        if len(potential_list) < 2: raise ValueError("Too few potentials")
        if len(potential_list) > 4: raise ValueError("Too many potentials")

        self.product_potentials.append(potential_list)

    def get_restraints(self):
        ret = []
        for c in self.children + self.group_sites:
            ret.extend( c.get_restraints() )

        return ret

    def get_bonds(self):
        ret = copy(self.bonds)
        for c in self.children:
            if isinstance(c,Parent): ret.extend( c.get_bonds() )
        if self.remove_duplicate_bonded_terms:
            return list(set(tuple(ret)))
        else:
            return ret

    def get_angles(self):
        ret = copy(self.angles)
        for c in self.children:
            if isinstance(c,Parent): ret.extend( c.get_angles() )
        if self.remove_duplicate_bonded_terms:
            return list(set(tuple(ret)))
        else:
            return ret

    def get_dihedrals(self):
        ret = copy(self.dihedrals)
        for c in self.children:
            if isinstance(c,Parent): ret.extend( c.get_dihedrals() )
        if self.remove_duplicate_bonded_terms:
            return list(set(tuple(ret)))
        else:
            return ret

    def get_bondXY(self):
        ret = copy(self.bondXY)
        for c in self.children:
            if isinstance(c,Parent): ret.extend( c.get_bondXY() )
        if self.remove_duplicate_bonded_terms:
            return list(set(tuple(ret)))
        else:
            return ret

    def get_vector_angles(self):
        ret = self.vector_angles
        for c in self.children:
            if isinstance(c,Parent): ret.extend( c.vector_angles() )
        if self.remove_duplicate_bonded_terms:
            return list(set(tuple(ret)))
        else:
            return ret

    def get_impropers(self):
        ret = copy(self.impropers)
        for c in self.children:
            if isinstance(c,Parent): ret.extend( c.get_impropers() )
        if self.remove_duplicate_bonded_terms:
            return list(set(tuple(ret)))
        else:
            return ret

    def get_exclusions(self):
        ret = copy(self.exclusions)
        for c in self.children:
            if isinstance(c,Parent): ret.extend( c.get_exclusions() )
        if self.remove_duplicate_bonded_terms:
            return list(set(tuple(ret)))
        else:
            return ret

    def get_vector_angles(self):
        ret = copy(self.vector_angles)
        for c in self.children:
            if isinstance(c,Parent): ret.extend( c.get_vector_angles() )
        if self.remove_duplicate_bonded_terms:
            return list(set(tuple(ret)))
        else:
            return ret

    def get_bond_angles(self):
        ret = copy(self.bond_angles)
        for c in self.children:
            if isinstance(c,Parent): ret.extend( c.get_bond_angles() )
        if self.remove_duplicate_bonded_terms:
            return list(set(tuple(ret)))
        else:
            return ret

    def get_product_potentials(self):
        ret = copy(self.product_potentials)
        for c in self.children:
            if isinstance(c,Parent): ret.extend( c.get_product_potentials() )
        if self.remove_duplicate_bonded_terms:
            return list(set(tuple(ret)))
        else:
            return ret

    def _get_bond_potentials(self):
        bonds =  [b for i,j,b,ex,sw in self.get_bonds()]
        bondXY =  [b for i,j,b,ax,ex,sw in self.get_bondXY()]
        bondangles1 = [b[1] for i,j,k,l,b,sw in self.get_bond_angles()]
        return list(set( tuple(bonds+bondXY+bondangles1) ))

    def _get_angle_potentials(self):
        angles =  [b for i,j,k,b,sw in self.get_angles()]
        bondangles1 = [b[0] for i,j,k,l,b,sw in self.get_bond_angles()]
        bondangles2 = [b[2] for i,j,k,l,b,sw in self.get_bond_angles()]
        return list(set( tuple(angles+bondangles1+bondangles2) ))


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

    def __tree_call_depth(self, function, post_function=None, start_depth=0, end_depth=None, function_args=None, function_kwargs=None, _root_call=True, *args, **kwargs):
        """
        Return a generator that returns the result of `function`
        applied to each element of the tree, traversed in a
        depth-first manner. After all children of a node have been
        processed, optionally apply `post_function` to each element.
        """
        if function_args is not None: args = function_args
        if function_kwargs is not None: kwargs = function_kwargs

        ## Gather distance from root and leaves if needed
        if _root_call and not (start_depth == 0 and end_depth is None):
            self._tree_depth = 0
            def assign_depth(x):
                if x is not self:
                    x._tree_depth = x.parent._tree_depth+1 if x.parent is not None else 0
                devlogger.debug(f'assigned depth {x._tree_depth} to {x}')
            def assign_depth_from_end(x):
                if isinstance(x, Parent):
                    x._tree_depth_end = min([0]+[c._tree_depth_end for c in x.children]) + 1
                else:
                    x._tree_depth_end = 0
                devlogger.debug(f'assigned end depth {x._tree_depth_end} to {x}')
            for result in self.__tree_call_depth(assign_depth, assign_depth_from_end, _root_call = False):
                pass

        ## Create temporary function to check if current item is in specified range
        if not (start_depth == 0 and end_depth is None):
            def _is_in_range(item):
                depth = item._tree_depth
                depth_from_end = item._tree_depth_end
                in_range = ((start_depth >= 0 and depth >= start_depth) or \
                            (start_depth < 0 and depth_from_end < -start_depth))
                if in_range and end_depth is not None:
                    in_range = ((end_depth >= 0 and depth < end_depth) or \
                                (end_depth < 0 and depth_from_end >= -end_depth))
                devlogger.debug(f'  {item = }, {start_depth = }, {end_depth = }, {depth = }, {depth_from_end = }, {in_range = }')
                return in_range
        else:
            def _is_in_range(item):
                return True

        ## Traverse tree
        if _is_in_range(self): yield function(self,*args,**kwargs)
        for x in self.children:
            if isinstance(x,Parent):
                if isinstance(x,Clone) and not isinstance(x.get_original_recursively(),Parent):
                    if not _is_in_range(x): continue
                    yield function(x,*args,**kwargs)
                    if post_function is not None:
                        post_function(x)
                else:
                    for result in x.__tree_call_depth(function, post_function, start_depth, end_depth, function_args=None, function_kwargs=None, _root_call=False, *args, **kwargs):
                        yield result
                    if post_function is not None:
                        post_function(x)
            else:
                if not _is_in_range(x): continue
                yield function(x,*args,**kwargs)
                if post_function is not None:
                    post_function(x,*args,**kwargs)

        if post_function is not None and  _is_in_range(self):
            post_function(self,*args,**kwargs)

    def clear_bonds(self):
        def fn(x): x.bonds = []
        for x in self.__tree_call_depth(fn, end_depth=-1): pass

    def clear_angles(self):
        def fn(x): x.angles = []
        for x in self.__tree_call_depth(fn, end_depth=-1): pass

    def clear_dihedrals(self):
        def fn(x): x.dihedrals = []
        for x in self.__tree_call_depth(fn, end_depth=-1): pass

    def clear_restraints(self):
        def fn(x): x.restraints = []
        for x in self.__tree_call_depth(fn, end_depth=-1): pass

    def __iter__(self):
        fn = lambda x: x
        return self.__tree_call_depth(fn, start_depth=-1)

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
    """
    A class that represents a child object which can inherit attributes from a parent.
    This class establishes a parent-child relationship where attributes not found in the child 
    can be looked up from the parent. This implementation allows for a form of delegation where 
    attribute access is forwarded to the parent object.

    Parameters:
        parent (Parent, optional): The parent object that this child is associated with.
                                  When set, the child registers itself with the parent.
                                  Defaults to None.
    
    Note:
        - When a parent is provided, it must be an instance of the Parent class.
        - Method lookups from the parent are explicitly prevented.
        - Certain attributes like 'parent' are excluded from the parent lookup mechanism.
    """
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

        ## Skip certain attributes from search
        excluded_attributes = ['parent']
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
    """
    Class that holds common attributes that particles can point to.
    ParticleType serves as a template for properties shared across multiple particles.
    It supports inheritance through parent types, attribute lookup, and provides mechanisms
    for comparing particle types.

    Parameters:
        excludedAttributes (tuple): Parameters that are not inherited from parent types.
        name (str): Unique identifier for this particle type.
        charge (float): Electric charge of the particle. Defaults to 0.
        mass (float, optional): Mass of the particle.
        diffusivity (float, optional): Diffusion coefficient of the particle.
        damping_coefficient (float, optional): Damping coefficient for dynamics.
        parent (ParticleType, optional): Parent type to inherit properties from.

    Note:
        - When a parent is specified, all non-excluded attributes are inherited.
        - Particle types with the same name must have identical properties.
        - The class implements custom copy behavior to prevent unnecessary duplication.
    Class that hold common attributes that particles can point to"""

    excludedAttributes = ("idx","type_",
                          "position",
                          "orientation",
                          "children",
                          "name",
                          "switch_type",
                          "parent", "excludedAttributes",
    )

    def __init__(self, name, charge=0, mass=None, diffusivity=None,
                 damping_coefficient=None, parent=None,
                 rigid_body_potentials=tuple(), switch_type = None, **kwargs):

        """ Parent type is used to fall back on for nonbonded
        interactions if this type is not specifically referenced """

        if parent is not None:
            for k,v in parent.__dict__.items():
                if k not in ParticleType.excludedAttributes:
                    self.__dict__[k] = v
            assert( type(parent) == type(self) )

        # if diffusivity is None:
        #     assert( (damping_coefficient is not None) and (mass is not None) )

        ## TODO: make most attributes @property
        self.name   = name
        self.charge = charge
        if mass is not None: self.mass = mass
        if damping_coefficient is not None: self.damping_coefficient = damping_coefficient
        if diffusivity is not None: self.diffusivity = diffusivity
        self.parent = parent
        self.rigid_body_potentials = rigid_body_potentials
        self.switch_type = switch_type
        devlogger.debug(f'Created {type(self)} {name} @ {hex(id(self))}')
        
        for key in ParticleType.excludedAttributes:
            assert( key not in kwargs )

        for key,val in kwargs.items():
            self.__dict__[key] = val

    def is_same_type(self, other, consider_parent_generations=-1):
        """
        Checks if this object is of the same type as another object.
        
        This method compares two objects to determine if they are of the same type.
        Objects are considered the same type if they are equal or if one is the parent
        of the other (when consider_parents=True).
        
        Parameters
        ----------
        other : same type as self
            The object to compare with
        consider_parents : bool, default=True
            If True, objects are considered the same type if one is the parent of the other
            
        Returns
        -------
        bool
            True if objects are considered of the same type, False otherwise
            
        Note
        -----
        The method assumes both objects are of the same Python class type.
        """
        assert( type(other) == type(self) )
        if self == other:
            return True
        elif consider_parent_generations != 0:
            if self.parent is not None:
               return self.parent.is_same_type(other, consider_parent_generations= consider_parent_generations-1)
        else:
            return False

    def add_grid_potential(self, gridfile, scale=1, boundary_condition='dirichlet'):
        """
        Adds a grid potential to the model.

        This method appends a new grid potential to the model's list of grid potentials.
        The grid potential is defined by a grid file, a scaling factor, and boundary conditions.

        Parameters
        ----------
        gridfile : str
            Path to the file containing the grid potential data.
        scale : float, default=1
            Scaling factor to apply to the grid potential.
        boundary_condition : str, default='dirichlet'
            The type of boundary condition to use for the grid potential.
            Must be one of 'dirichlet', 'neumann', or 'periodic'.

        Raises
        ------
        ValueError
            If the boundary_condition is not one of 'dirichlet', 'neumann', or 'periodic'.

        Returns
        -------
        None
        """
        if boundary_condition not in ('dirichlet','neumann','periodic'):
            raise ValueError(f'Unrecognized grid boundary condition "{boundary_condition}"; should be one of "dirichlet", "neumann" or "periodic".')
        self.grid_potentials = getattr(self, 'grid_potentials', []) + [(gridfile,scale,boundary_condition)]
        
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
        
    def _hash_key(self):

        l = [str(type(self)), self.name, self.charge]
        for keyval in sorted(self.__dict__.items()):
            if isinstance(keyval[1], list): keyval = (keyval[0],tuple(keyval[1]))
            l.extend(keyval)
        return tuple(l)

    def __hash__(self):
        return hash(self._hash_key())
    
    def _equal_check(a,b):
        if a.name == b.name:
            if a._hash_key() != b._hash_key():
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

class RigidBodyType(ParticleType):
    """Class that holds common attributes for RigidBody objects.
    This class extends ParticleType to represent rigid bodies that can have
    attached particles, orientation, and rotational dynamics.

    Parameters:
        name (str): Name identifier for the rigid body type.
        parent (ParticleType, optional): Parent type to fall back on for nonbonded interactions.
        moment_of_inertia (float or array-like, optional): Moment of inertia tensor for the rigid body.
        rotational_diffusivity (float or array-like, optional): Rotational diffusivity coefficient.
        rotational_damping_coefficient (float or array-like, optional): Rotational damping coefficient.
        attached_particles (tuple or list): Particles attached to this rigid body.
        potential_grids (tuple): Collection of potential grid definitions, each with length 2 or 3.
        charge_grids (tuple): Collection of charge grid definitions, each with length 2 or 3.
        pmf_grids (tuple): Collection of pmf grid definitions, each with length 2 or 3.

    Note:
        If rotational_diffusivity is not provided, both moment_of_inertia and 
        rotational_damping_coefficient must be specified.
    """

    """Class that holds common attributes for RigidBody objects"""

    def __init__(self, name, parent=None, moment_of_inertia = None,
                 rotational_diffusivity = None,
                 rotational_damping_coefficient = None,
                 attached_particles=tuple(), potential_grids=tuple(),
                 charge_grids=tuple(), pmf_grids=tuple(), **kwargs):

        """ Parent type is used to fall back on for nonbonded
        interactions if this type is not specifically referenced """

        if rotational_diffusivity is None:
            assert( (rotational_damping_coefficient is not None) and (moment_of_inertia is not None) )
        
        for _grids in (potential_grids,charge_grids,pmf_grids):
                for val in _grids:
                    assert( len(val) in (2,3) ) #                 
                
        ParticleType.__init__(self, name, parent=parent,
                              moment_of_inertia = moment_of_inertia,
                              rotational_diffusivity=rotational_diffusivity,
                              rotational_damping_coefficient = rotational_damping_coefficient,
                              potential_grids = potential_grids,
                              charge_grids = charge_grids,
                              pmf_grids = pmf_grids,
                              **kwargs)

        self.attached_particles = []
        for p in attached_particles:
            self.attach_particle(p)

    def attach_particle(self, particle):
        """ The particle argument must be a PointParticle. The position/orientation of the attached particle/group is in the RigidBody frame. """
        
        if particle.parent is not None:
            raise ValueError('RigidBody-attached particles are not allowed to have a parent')
        self.attached_particles.append( particle )
        
    def _equal_check(a,b):
        if a.name == b.name:
            if a._hash_key() != b._hash_key():
                raise Exception("Two different RigidBodyTypes have same 'name' attribute")    

class PointParticle(Transformable, Child):
    """
    A class representing a point particle in a simulation system.
    This class inherits from both Transformable and Child base classes, allowing
    it to be positioned in space and exist in a parent-child hierarchy.

    Parameters:
        type_ (ParticleType): Type definition of the particle.
        idx (int): Index of the particle in the system, default is None.
        name (str): Name identifier for the particle, default is "A".
        counter (int): Counter for operations on this particle, initialized to 0.
        restraints (list): List of restraints applied to this particle.
        rigid (bool): Flag indicating whether the particle is rigid, default is False.
    """
    def __init__(self, type_, position, name="A", switch_schedule=None, **kwargs):
        parent = None
        if 'parent' in kwargs:
            parent = kwargs['parent']
        Child.__init__(self, parent=parent)
        Transformable.__init__(self,position)

        self.type_    = type_                
        self.idx     = None
        self.name = name
        self.switch_schedule = [] if switch_schedule is None else switch_schedule
        self.counter = 0
        self.restraints = []
        self.rigid = False
        
        for key,val in kwargs.items():
            self.__dict__[key] = val
        
    def add_restraint(self, restraint):
        """
        Add a restraint to the model's restraint list.

        Parameters
        ----------
        restraint : Restraint
            The restraint object to add to the model.

        Note
        -----
        TODO: Determine how to handle duplicating and cloning bonds.
        """
        self.restraints.append( restraint )

    def add_grid_potential(self, gridfile, scale=1, boundary_condition='dirichlet'):
        """
        Add a grid-based potential to the particle type.
        
        This method creates a new particle type derived from the current one and adds 
        a grid-based potential to it. The grid potential is loaded from the specified file.
        After adding the potential, the particle's type is updated to the new type.
        
        Parameters
        ----------
        gridfile : str
            Path to the grid file in .dx format containing the potential data.
        scale : float, optional
            Scaling factor applied to the potential values. Default is 1.
        boundary_condition : str, optional
            Type of boundary condition to use for the grid potential.
            Options are 'dirichlet' (default) or other boundary types supported by the system.
        
        Returns
        -------
        None
            The method updates the particle's type in place.
        
        Note
        -----
        This method will create a copy of the particle's type if it already has a parent,
        or create a new derived type if it doesn't.
        """
        t0 = self.type_
        name = f'{t0.name}_g_{Path(gridfile).stem}_s_{scale}'
        if t0.parent is not None:
            t = copy(t0)
            t.name = name
        else:
            # TODO: REMOVE LINE: t = ParticleType(name, parent=t0)
             t = type(t0)(name, parent=t0)
        t.add_grid_potential(gridfile, scale=scale, boundary_condition=boundary_condition)
        self.type_ = t
        self._clear_types()
        
    def get_restraints(self):
        """
        Get all restraints associated with this model.

        Returns:
            list[tuple]: A list of tuples, where each tuple contains:
                - First element: this model instance
                - Second element: a restraint object associated with this model
        """
        return [(self,r) for r in self.restraints]

    def duplicate(self):
        """
        Create a deep copy of the current object.

        This method creates a completely independent copy of the object, using the `deepcopy` function.
        Any modifications to the returned object will not affect the original object.

        Returns:
            A new instance of the same class with identical attribute values.
        """
        new = deepcopy(self)
        return new

    def add_switch_step(self, step, value):
        if step < 0:
            raise ValueError(f'Particle type switching step should be non-negative')

        if len(self.switch_schedule) > 0 and self.switch_schedule[-1][0] >= step:
            raise ValueError(f'Particle type switching step mut occur later than previous step')

        if step != int(step):
            raise ValueError('Particle type switching step must be an integer')

        if value != float(value) or value < 0 or value > 1:
            raise ValueError('Particle type switching value must be number between zero and one')

        self.switch_schedule.append((step,value))

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
            if mass is None: raise
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

    def __repr__(self):
        return f'<{__name__}.{self.__class__.__name__} "{self.name}" of {self.type_}>'

class RigidBody(PointParticle):
    """
    Represents a rigid body in a physical simulation.
    A rigid body is a collection of particles that maintain fixed positions relative to each other.
    It inherits from PointParticle and implements transformation capabilities to move and rotate as a single unit.

    Parameters
    ----------
    type_ : RigidBodyType
        The type definition for this rigid body.
    idx : int or None
        Index identifier, assigned when added to a simulation.
    name : str
        Name identifier for this rigid body.
    counter : int
        Counter value, initialized at 0.
    restraints : list
        List of restraints applied to this rigid body.
    rigid : bool
        Boolean flag indicating if the object is rigid, always True for this class.
    attached_particles : list
        Copied particles attached to the rigid body. 
    **kwargs : 
        Additional keyword arguments to set as attributes.

    Note
    ----------
    TODO: for attached_particles, it should be possible to uniquely apply bonds/angles etc to these particles, but their types should be fixed or otherwise unified among rbs; 
    here we are copying them simply so that they can recieve and index and be used in bonded potentials and group sites

    """

    def __init__(self, type_, position, orientation, name="A", attached_particles=tuple(), **kwargs):
        parent = None
        if 'parent' in kwargs:
            parent = kwargs['parent']
        Child.__init__(self, parent=parent)
        Transformable.__init__(self,position, orientation)

        if type(type_) != RigidBodyType:
            raise ValueError(f'Attempted to create a RigidBody object from an invalid type {type_}')

        self.type_    = type_                
        self.idx     = None
        self.name = name
        self.counter = 0
        self.restraints = []
        self.rigid = True

        self.attached_particles = [copy(p) for p in type_.attached_particles]
        
        for key,val in kwargs.items():
            self.__dict__[key] = val
        
    def add_restraint(self, restraint):
        raise NotImplementedError('Harmonic restraints are not yet supported for rigid bodies; consider implementing this by attaching a dummy particle')
        ## TODO: how to handle duplicating and cloning bonds
        self.restraints.append( restraint )

    def get_restraints(self):
        ret = [(self,r) for r in self.restraints]
        for p in self.attached_particles:
            ret.extend( p.get_restraints() )
        return ret

    def duplicate(self):
        new = deepcopy(self)
        return new

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
    """
    A class representing a group of objects that can be transformed, have children, and be a child.
    The Group class inherits from Transformable, Parent, and Child, combining the functionality
    of all three classes. It allows for hierarchical grouping of objects, where each group can
    have a position, orientation, parent, and children.

    Parameters:
        name (str, optional): The name of the group.
        children (list, optional): A list of child objects belonging to this group.
        parent (Parent, optional): The parent object of this group.
        position (numpy.ndarray): The 3D position of the group, defaults to origin (0,0,0).
        orientation (numpy.ndarray): The 3x3 rotation matrix representing the orientation,
            defaults to identity matrix.
        isClone (bool): Flag indicating whether this group is a clone, defaults to False.
        remove_duplicate_bonded_terms (bool): Whether to remove duplicate bonded terms when
            combining children, defaults to False.
            
    Methods:
        clone(): Creates a clone of this group.
        duplicate(): Creates a deep copy of this group, preserving parent-child relationships.
    """

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

    ## TODO override deepcopy so parent can be excluded from copying?
        
    # def __getstate__(self):
    #     return (self.children, self.parent, self.position, self.orientation)

    # def __setstate__(self, state):
    #     self.children, self.parent, self.position, self.orientation = state


class GroupSite:
    """
    Class to represent a collection of particles that can be used by bond potentials in arbdmodel.
    This class groups multiple particles together and provides methods to calculate their center
    and manage restraints applied to the group.

    Parameters
    ----------
    particles : list
        List of particle objects to be included in the group.
    idx : int or None
        Identifier for the group site, default is None.
    restraints : list
        List of restraints applied to the group site.

    Methods
    -------
    get_center()
        Calculate the center position of the group by averaging particle positions.
    add_restraint(restraint)
        Add a restraint to the group site.
    get_restraints()
        Get all restraints associated with this group site as tuples of (site, restraint).

    Note
    -----
    Currently does not support weighted particles (weights parameter in __init__).
    """
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
        self.restraints.append(restraint)
        
    def get_restraints(self):
        return [(self, r) for r in self.restraints]

