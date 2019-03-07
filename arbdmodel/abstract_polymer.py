from pathlib import Path
from copy import copy, deepcopy

import numpy as np
from scipy import interpolate

from . import ArbdModel
from .coords import rotationAboutAxis, quaternion_from_matrix, quaternion_to_matrix



"""
TODO:
 - test for large systems
 - rework Location class 
 - remove recursive calls
 - document
 - develop unit test suite
"""

# class ParticleNotConnectedError(Exception):
#     pass

class Location():
    """ Site for connection within an object """
    def __init__(self, parent, contour_position, type_, on_fwd_strand = True):
        ## TODO: remove cyclic references(?)
        assert( isinstance(parent, ConnectableElement) )
        self.parent = parent
        self.contour_position = contour_position  # represents position along contour length in polymer
        self.type_ = type_
        # self.particle = None
        self.connection = None

    def get_connected_location(self):
        if self.connection is None:
            return None
        else:
            return self.connection.other(self)

    def set_connection(self, connection):
        self.connection = connection # TODO weakref? 

    def get_monomer_index(self):
        try:
            idx = self.parent.contour_to_monomer_index(self.contour_position, nearest_monomer=True)
        except:
            if self.contour_position == 0:
                idx = 0
            elif self.contour_position == 1:
                idx = self.parent.num_monomers-1
            else:
                raise
        return idx

    def __repr__(self):
        if self.on_fwd_strand:
            on_fwd = "on_fwd_strand"
        else:
            on_fwd = "on_rev_strand"
        return "<Location {}.{}[{:.2f},{:d}]>".format( self.parent.name, self.type_, self.contour_position, self.on_fwd_strand)
        
class Connection():
    """ Class for connecting two elements """
    def __init__(self, A, B, type_ = None):
        assert( isinstance(A,Location) )
        assert( isinstance(B,Location) )
        self.A = A
        self.B = B
        self.type_ = type_
        
    def other(self, location):
        if location is self.A:
            return self.B
        elif location is self.B:
            return self.A
        else:
            raise ValueError

    def delete(self):
        self.A.parent.connections.remove(self)
        if self.B.parent is not self.A.parent:
            self.B.parent.connections.remove(self)
        self.A.connection = None
        self.B.connection = None

    def __repr__(self):
        return "<Connection {}--{}--{}]>".format( self.A, self.type_, self.B )
        
class ConnectableElement():
    """ Abstract base class """
    def __init__(self, connection_locations=None, connections=None):
        if connection_locations is None: connection_locations = []
        if connections is None: connections = []

        ## TODO decide on names
        self.locations = self.connection_locations = connection_locations
        self.connections = connections

    def get_locations(self, type_=None, exclude=()):
        locs = [l for l in self.connection_locations if (type_ is None or l.type_ == type_) and l.type_ not in exclude]
        counter = dict()
        for l in locs:
            if l in counter:
                counter[l] += 1
            else:
                counter[l] = 1
        assert( np.all( [counter[l] == 1 for l in locs] ) )
        return locs

    def get_location_at(self, contour_position, on_fwd_strand=True, new_type="crossover"):
        loc = None
        if (self.num_monomers == 1):
            ## Assumes that intrahelical connections have been made before crossovers
            for l in self.locations:
                if l.on_fwd_strand == on_fwd_strand and l.connection is None:
                    assert(loc is None)
                    loc = l
            # assert( loc is not None )
        else:
            for l in self.locations:
                if l.contour_position == contour_position and l.on_fwd_strand == on_fwd_strand:
                    assert(loc is None)
                    loc = l
        if loc is None:
            loc = Location( self, contour_position=contour_position, type_=new_type, on_fwd_strand=on_fwd_strand )
        return loc

    def get_connections_and_locations(self, connection_type=None, exclude=()):
        """ Returns a list with each entry of the form:
            connection, location_in_self, location_in_other """
        type_ = connection_type
        ret = []
        for c in self.connections:
            if (type_ is None or c.type_ == type_) and c.type_ not in exclude:
                if   c.A.parent is self:
                    ret.append( [c, c.A, c.B] )
                elif c.B.parent is self:
                    ret.append( [c, c.B, c.A] )
                else:
                    import pdb
                    pdb.set_trace()
                    raise Exception("Object contains connection that fails to refer to object")
        return ret

    def _connect(self, other, connection):
        ## TODO fix circular references        
        A,B = [connection.A, connection.B]
            
        A.connection = B.connection = connection
        self.connections.append(connection)
        if other is not self:
            other.connections.append(connection)
        else:
            raise NotImplementedError("Objects cannot yet be connected to themselves; if you are attempting to make a circular object, try breaking the object into multiple components")
        l = A.parent.locations
        if A not in l: l.append(A)
        l = B.parent.locations
        if B not in l: l.append(B)


class PolymerSection(ConnectableElement):
    """ Base class that describes a linear section of a polymer """

    def __init__(self, name, num_monomers,
                 monomer_length,
                 start_position = None,
                 end_position = None, 
                 parent = None,
                 **kwargs):
        
        ConnectableElement.__init__(self, connection_locations=[], connections=[])

        if 'segname' not in kwargs:
            self.segname = name

        self.num_monomers = int(num_monomers)
        self.monomer_length = monomer_length
        
        if start_position is None: start_position = np.array((0,0,0))

        if end_position is None:
            end_position = np.array((0,0,self.monomer_length*num_monomers)) + start_position
        self.start_position = start_position
        self.end_position = end_position

        self.start_orientation = None

        ## Set up interpolation for positions
        self._set_splines_from_ends()

        # self.sequence = None

    def __repr__(self):
        return "<{} {}[{:d}]>".format( type(self), self.name, self.num_monomers )

    def set_splines(self, contours, coords):
        tck, u = interpolate.splprep( coords.T, u=contours, s=0, k=1)
        self.position_spline_params = (tck,u)

    def set_orientation_splines(self, contours, quaternions):
        tck, u = interpolate.splprep( quaternions.T, u=contours, s=0, k=1)
        self.quaternion_spline_params = (tck,u)

    def get_center(self):
        tck, u = self.position_spline_params
        return np.mean(self.contour_to_position(u), axis=0)

    def _get_location_positions(self):
        return [self.contour_to_monomer_index(l.contour_position) for l in self.locations]

    def insert_monomers(self, at_monomer: int, num_monomer: int, seq=tuple()):
        assert(np.isclose(np.around(num_monomer),num_monomer))
        if at_monomer < 0:
            raise ValueError("Attempted to insert DNA into {} at a negative location".format(self))
        if at_monomer > self.num_monomer-1:
            raise ValueError("Attempted to insert DNA into {} at beyond the end of the Polymer".format(self))
        if num_monomers < 0:
            raise ValueError("Attempted to insert DNA a negative amount of DNA into {}".format(self))

        num_monomers = np.around(num_monomer)
        monomer_positions = self._get_location_positions()
        new_monomer_positions = [p if p <= at_monomer else p+num_monomers for p in monomer_positions]

        ## TODO: handle sequence

        self.num_monomers = self.num_monomer+num_monomer

        for l,p in zip(self.locations, new_monomer_positions):
            l.contour_position = self.monomer_index_to_contour(p)

    def remove_monomers(self, first_monomer: int, last_monomer: int):
        """ Removes nucleotides between first_monomer and last_monomer, inclusive """
        assert(np.isclose(np.around(first_monomer),first_monomer))
        assert(np.isclose(np.around(last_monomer),last_monomer))
        tmp = min((first_monomer,last_monomer))
        last_monomer = max((first_monomer,last_monomer))
        fist_monomer = tmp

        if first_monomer < 0 or first_monomer > self.num_monomer-2:
            raise ValueError("Attempted to remove DNA from {} starting at an invalid location {}".format(self, first_monomer))
        if last_monomer < 1 or last_monomer > self.num_monomer-1:
            raise ValueError("Attempted to remove DNA from {} ending at an invalid location {}".format(self, last_monomer))
        if first_monomer == last_monomer:
            return

        first_monomer = np.around(first_monomer)
        last_monomer = np.around(last_monomer)

        monomer_positions = self._get_location_positions()

        bad_locations = list(filter(lambda p: p >= first_monomer and p <= last_monomer, monomer_positions))
        if len(bad_locations) > 0:
            raise Exception("Attempted to remove DNA containing locations {} from {} between {} and {}".format(bad_locations,self,first_monomer,last_monomer))

        removed_monomer = last_monomer-first_monomer+1
        new_monomer_positions = [p if p <= last_monomer else p-removed_monomer for p in monomer_positions]
        num_monomers = self.num_monomer-removed_monomer

        if self.sequence is not None and len(self.sequence) == self.num_monomer:
            self.sequence = [s for s,i in zip(self.sequence,range(self.num_monomer)) 
                                if i < first_monomer or i > last_monomer]
            assert( len(self.sequence) == num_monomers )

        self.num_monomers = num_monomers

        for l,p in zip(self.locations, new_monomer_positions):
            l.contour_position = self.monomer_position_to_contour(p)

    def __filter_contours(contours, positions, position_filter, contour_filter):
        u = contours
        r = positions

        ## Filter
        ids = list(range(len(u)))
        if contour_filter is not None:
            ids = list(filter(lambda i: contour_filter(u[i]), ids))
        if position_filter is not None:
            ids = list(filter(lambda i: position_filter(r[i,:]), ids))
        return ids

    def translate(self, translation_vector, position_filter=None, contour_filter=None):
        dr = np.array(translation_vector)
        tck, u = self.position_spline_params
        r = self.contour_to_position(u)

        ids = PolymerSection.__filter_contours(u, r, position_filter, contour_filter)
        if len(ids) == 0: return

        ## Translate
        r[ids,:] = r[ids,:] + dr[np.newaxis,:]
        self.set_splines(u,r)

    def rotate(self, rotation_matrix, about=None, position_filter=None, contour_filter=None):
        tck, u = self.position_spline_params
        r = self.contour_to_position(u)

        ids = PolymerSection.__filter_contours(u, r, position_filter, contour_filter)
        if len(ids) == 0: return

        if about is None:
            ## TODO: do this more efficiently
            r[ids,:] = np.array([rotation_matrix.dot(r[i,:]) for i in ids])
        else:
            dr = np.array(about)
            ## TODO: do this more efficiently
            r[ids,:] = np.array([rotation_matrix.dot(r[i,:]-dr) + dr for i in ids])

        self.set_splines(u,r)

        if self.quaternion_spline_params is not None:
            ## TODO: performance: don't shift between quaternion and matrix representations so much
            tck, u = self.quaternion_spline_params
            orientations = [self.contour_to_orientation(v) for v in u]
            for i in ids:
                orientations[i,:] = rotation_matrix.dot(orientations[i])
            quats = [quaternion_from_matrix(o) for o in orientations]
            self.set_orientation_splines(u, quats)

    def _set_splines_from_ends(self, resolution=4):
        self.quaternion_spline_params = None
        r0 = np.array(self.start_position)[np.newaxis,:]
        r1 = np.array(self.end_position)[np.newaxis,:]
        u = np.linspace(0,1, max(3,self.num_monomers//int(resolution)))
        s = u[:,np.newaxis]
        coords = (1-s)*r0 + s*r1
        self.set_splines(u, coords)

    def contour_to_monomer_index(self, contour_pos, nearest_monomer=False):
        idx = contour_pos*(self.num_monomer) - 0.5
        if nearest_monomer:
            assert( np.isclose(np.around(idx),idx) )
            idx = np.around(idx)
        return idx

    def monomer_index_to_contour(self,monomer_pos):
        return (monomer_pos+0.5)/(self.num_monomers)

    def contour_to_position(self,s):
        p = interpolate.splev( s, self.position_spline_params[0] )
        if len(p) > 1: p = np.array(p).T
        return p

    def contour_to_tangent(self,s):
        t = interpolate.splev( s, self.position_spline_params[0], der=1 )
        t = (t / np.linalg.norm(t,axis=0))
        return t.T
        

    def contour_to_orientation(self,s):
        assert( isinstance(s,float) or isinstance(s,int) or len(s) == 1 )   # TODO make vectorized version

        if self.quaternion_spline_params is None:
            axis = self.contour_to_tangent(s)
            axis = axis / np.linalg.norm(axis)
            rotAxis = np.cross(axis,np.array((0,0,1)))
            rotAxisL = np.linalg.norm(rotAxis)
            zAxis = np.array((0,0,1))

            if rotAxisL > 0.001:
                theta = np.arcsin(rotAxisL) * 180/np.pi
                if axis.dot(zAxis) < 0: theta = 180-theta
                orientation0 = rotationAboutAxis( rotAxis/rotAxisL, theta, normalizeAxis=False ).T
            else:
                orientation0 = np.eye(3) if axis.dot(zAxis) > 0 else \
                               rotationAboutAxis( np.array((1,0,0)), 180, normalizeAxis=False )
            if self.start_orientation is not None:
                orientation0 = orientation0.dot(self.start_orientation)

            # orientation = rotationAboutAxis( axis, self.twist_per_nt*self.contour_to_monomer_index(s), normalizeAxis=False )
            # orientation = orientation.dot(orientation0)
            orientation = orientation0
        else:
            q = interpolate.splev( s, self.quaternion_spline_params[0] )
            if len(q) > 1: q = np.array(q).T # TODO: is this needed?
            orientation = quaternion_to_matrix(q)

        return orientation

    def get_contour_sorted_connections_and_locations(self,type_):
        sort_fn = lambda c: c[1].contour_position
        cl = self.get_connections_and_locations(type_)
        return sorted(cl, key=sort_fn)
    
    def add_location(self, nt, type_, on_fwd_strand=True):
        ## Create location if needed, add to polymer
        c = self.monomer_index_to_contour(nt)
        assert(c >= 0 and c <= 1)
        # TODO? loc = self.Location( contour_position=c, type_=type_, on_fwd_strand=is_fwd )
        loc = Location( self, contour_position=c, type_=type_, on_fwd_strand=on_fwd_strand )
        self.locations.append(loc)

    def iterate_connections_and_locations(self, reverse=False):
        ## connections to other polymers
        cl = self.get_contour_sorted_connections_and_locations()
        if reverse:
            cl = cl[::-1]
            
        for c in cl:
            yield c
            
class AbstractPolymerGroup():
    def __init__(self, polymers=[],
                 **kwargs):

        """ Simulation system parameters """
        ## TODO decide whether to move these elsewhere
        # self.dimensions = dimensions
        # self.temperature = temperature
        # self.timestep = timestep
        # self.cutoff = cutoff
        # self.decomposition_period = decomposition_period
        # self.pairlist_distance = pairlist_distance
        # self.nonbonded_resolution = nonbonded_resolution
        self.polymers = polymers

        self.grid_potentials = []

        # self.useNonbondedScheme( nbDnaScheme )

    def get_connections(self,type_=None,exclude=()):
        """ Find all connections in model, without double-counting """
        added=set()
        ret=[]
        for s in self.polymers:
            items = [e for e in s.get_connections_and_locations(type_,exclude=exclude) if e[0] not in added]
            added.update([e[0] for e in items])
            ret.extend( list(sorted(items,key=lambda x: x[1].contour_position)) )
        return ret
    
    def extend(self, other, copy=True, include_strands=False):
        assert( isinstance(other, PolymerSectionModel) )
        if copy:
            for s in other.polymers:
                self.polymers.append(deepcopy(s))
            if include_strands:
                for s in other.strands:
                    self.strands.append(deepcopy(s))
        else:
            for s in other.polymers:
                self.polymers.append(s)
            if include_strands:
                for s in other.strands:
                    self.strands.append(s)
        self._clear_beads()

    def update(self, polymer , copy=False):
        assert( isinstance(polymer, PolymerSection) )
        if copy:
            polymer = deepcopy(polymer)
        self.polymers.append(polymer)
    
    """ Operations on spline coordinates """
    def translate(self, translation_vector, position_filter=None):
        for s in self.polymers:
            s.translate(translation_vector, position_filter=position_filter)
        
    def rotate(self, rotation_matrix, about=None, position_filter=None):
        for s in self.polymers:
            s.rotate(rotation_matrix, about=about, position_filter=position_filter)

    def get_center(self, include_ssdna=False):
        if include_ssdna:
            polymers = self.polymers
        else:
            polymers = list(filter(lambda s: isinstance(s,DoubleStrandedPolymerSection), 
                                   self.polymers))
        centers = [s.get_center() for s in polymers]
        weights = [s.num_monomers*2 if isinstance(s,DoubleStrandedPolymerSection) else s.num_monomers for s in polymers]
        # centers,weights = [np.array(a) for a in (centers,weights)]
        return np.average( centers, axis=0, weights=weights)

    def dimensions_from_structure( self, padding_factor=1.5, isotropic=False ):
        positions = []
        for s in self.polymers:
            positions.append(s.contour_to_position(0))
            positions.append(s.contour_to_position(0.5))
            positions.append(s.contour_to_position(1))
        positions = np.array(positions)
        dx,dy,dz = [(np.max(positions[:,i])-np.min(positions[:,i])+30)*padding_factor for i in range(3)]
        if isotropic:
            dx = dy = dz = max((dx,dy,dz))
        return [dx,dy,dz]

    def add_grid_potential(self, grid_file, scale=1, per_monomer=True):
        grid_file = Path(grid_file)
        if not grid_file.is_file():
            raise ValueError("Grid file {} does not exist".format(grid_file))
        if not grid_file.is_absolute():
            grid_file = Path.cwd() / grid_file
        self.grid_potentials.append((grid_file,scale,per_,pmp,er))

    def vmd_tube_tcl(self, file_name="drawTubes.tcl", radius=5):
        with open(file_name, 'w') as tclFile:
            tclFile.write("## beginning TCL script \n")

            def draw_tube(polymer,radius_value=10, color="cyan", resolution=5):
                tclFile.write("## Tube being drawn... \n")
                
                contours = np.linspace(0,1, max(2,1+polymer.num_monomers//resolution) )
                rs = [polymer.contour_to_position(c) for c in contours]
               
                radius_value = str(radius_value)
                tclFile.write("graphics top color {} \n".format(str(color)))
                for i in range(len(rs)-2):
                    r0 = rs[i]
                    r1 = rs[i+1]
                    filled = "yes" if i in (0,len(rs)-2) else "no"
                    tclFile.write("graphics top cylinder {{ {} {} {} }} {{ {} {} {} }} radius {} resolution 30 filled {} \n".format(r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], str(radius_value), filled))
                    tclFile.write("graphics top sphere {{ {} {} {} }} radius {} resolution 30\n".format(r1[0], r1[1], r1[2], str(radius_value)))
                r0 = rs[-2]
                r0 = rs[-1]
                tclFile.write("graphics top cylinder {{ {} {} {} }} {{ {} {} {} }} radius {} resolution 30 filled yes \n".format(r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], str(radius_value)))

            ## material
            tclFile.write("graphics top materials on \n")
            tclFile.write("graphics top material AOEdgy \n")
            
            ## iterate through the model polymers
            for s in self.polymers:
                draw_tube(s,radius,"cyan")


    def vmd_cylinder_tcl(self, file_name="drawCylinders.tcl", radius=5):

        with open(file_name, 'w') as tclFile:
            tclFile.write("## beginning TCL script \n")
            def draw_cylinder(polymer,radius_value=10,color="cyan"):
                tclFile.write("## cylinder being drawn... \n")
                r0 = polymer.contour_to_position(0)
                r1 = polymer.contour_to_position(1)
                
                radius_value = str(radius_value)
                color = str(color)
                
                tclFile.write("graphics top color {} \n".format(color))
                tclFile.write("graphics top cylinder {{ {} {} {} }} {{ {} {} {} }} radius {} resolution 30 filled yes \n".format(r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], radius_value))

            ## material
            tclFile.write("graphics top materials on \n")
            tclFile.write("graphics top material AOEdgy \n")
            
            ## iterate through the model polymers
            for s in self.polymers:
                draw_cylinder(s,radius,"cyan")
