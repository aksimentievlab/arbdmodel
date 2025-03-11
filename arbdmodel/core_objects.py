# -*- coding: utf-8 -*-
import numpy as np
from inspect import ismethod
from copy import copy, deepcopy
from . import logger, get_resource_path,devlogger


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
        self.vector_angles = []
        self.impropers = []
        self.exclusions = []
        self.vector_angles = []
        self.bond_angles = []
        self.product_potentials = []
        self.group_sites = []
        
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

    def add_vector_angle(self, i,j,k,l, potential):
        assert( len(set((i,j,k,l))) == 4 )

        # beads = [b for b in self]
        # for b in (i,j,k,l): assert(b in beads)
        self.vector_angles.append( (i,j,k,l, potential) )

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

    def add_vector_angle(self, i,j,k,l, potential):
        assert( len(set((i,j,k,l))) >= 3 )
        self.vector_angles.append( (i,j,k,l, potential) )

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
        bonds =  [b for i,j,b,ex in self.get_bonds()]
        bondangles1 = [b[1] for i,j,k,l,b in self.get_bond_angles()]
        return list(set( tuple(bonds+bondangles1) ))

    def _get_angle_potentials(self):
        angles =  [b for i,j,k,b in self.get_angles()]
        bondangles1 = [b[0] for i,j,k,l,b in self.get_bond_angles()]
        bondangles2 = [b[2] for i,j,k,l,b in self.get_bond_angles()]
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
    """Class that hold common attributes that particles can point to"""

    excludedAttributes = ("idx","type_",
                          "position",
                          "orientation",
                          "children",
                          "name",
                          "parent", "excludedAttributes",
    )

    def __init__(self, name, charge=0, mass=None, diffusivity=None,
                 damping_coefficient=None, parent=None,
                 rigid_body_potentials=tuple(), **kwargs):

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
        devlogger.debug(f'Created {type(self)} {name} @ {hex(id(self))}')
        
        for key in ParticleType.excludedAttributes:
            assert( key not in kwargs )

        for key,val in kwargs.items():
            self.__dict__[key] = val

    def is_same_type(self, other, consider_parents=True):
        assert( type(other) == type(self) )
        if self == other:
            return True
        elif consider_parents:
            if self.parent is not None and self.parent == other:
                return True
            elif other.parent is not None and other.parent == self:
                return True
            # elif other.parent is not None and self.parent is not None and other.parent == self.parent:
            #     return True
        else:
            return False

    def add_grid_potential(self, gridfile, scale=1, boundary_condition='dirichlet'):
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

    def add_grid_potential(self, gridfile, scale=1, boundary_condition='dirichlet'):
        t0 = self.type_
        name = f'{t0.name}_g_{gridfile.replace(".dx","")}_s_{scale}'
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

        ## TODO: it should be possible to uniquely apply bonds/angles etc to these particles, but their types should be fixed or otherwise unified among rbs; here we are copying them simply so that they can recieve and index and be used in bonded potentials and group sites
        self.attached_particles = [copy(p) for p in type_.attached_particles]
        
        for key,val in kwargs.items():
            self.__dict__[key] = val
        
    def add_restraint(self, restraint):
        raise NotImplementedError('Harmonic restraints are not yet supported for rigid bodies; consider implementing this by attaching a dummy particle')
        ## TODO: how to handle duplicating and cloning bonds
        self.restraints.append( restraint )

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

    ## TODO override deepcopy so parent can be excluded from copying?
        
    # def __getstate__(self):
    #     return (self.children, self.parent, self.position, self.orientation)

    # def __setstate__(self, state):
    #     self.children, self.parent, self.position, self.orientation = state

#Not sure where should it go
class GroupSite:
    """ Class to represent a collection of particles that can be used by bond potentials. In arbdmodel only """
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

