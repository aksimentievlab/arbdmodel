import pdb
from pathlib import Path
import numpy as np
import random
from .model.arbdmodel import PointParticle, ParticleType, Group, ArbdModel
from .coords import rotationAboutAxis, quaternion_from_matrix, quaternion_to_matrix
from .model.nonbonded import *
from copy import copy, deepcopy
from .model.nbPot import nbDnaScheme

from scipy.special import erf
import scipy.optimize as opt
from scipy import interpolate

from .model.CanonicalNucleotideAtoms import canonicalNtFwd, canonicalNtRev, seqComplement
from .model.CanonicalNucleotideAtoms import enmTemplateHC, enmTemplateSQ, enmCorrectionsHC

from .model.spring_from_lp import k_angle as angle_spring_from_lp

# import pdb
"""
TODO:
 + fix handling of crossovers for atomic representation
 + map to atomic representation
    + add nicks
    + transform ssDNA nucleotides 
    - shrink ssDNA
    + shrink dsDNA backbone
    + make orientation continuous
    + sequence
    + handle circular dna
 + ensure crossover bead potentials aren't applied twice 
 + remove performance bottlenecks
 - test for large systems
 + assign sequence
 + ENM
 - rework Location class 
 - remove recursive calls
 - document
 - develop unit test suite
"""
class CircularDnaError(Exception):
    pass

class ParticleNotConnectedError(Exception):
    pass

class Location():
    """ Site for connection within an object """
    def __init__(self, container, address, type_, on_fwd_strand = True):
        ## TODO: remove cyclic references(?)
        self.container = container
        self.address = address  # represents position along contour length in segment
        # assert( type_ in ("end3","end5") ) # TODO remove or make conditional
        self.on_fwd_strand = on_fwd_strand
        self.type_ = type_
        self.particle = None
        self.connection = None
        self.is_3prime_side_of_connection = None

        self.prev_in_strand = None
        self.next_in_strand = None
        
        self.combine = None     # some locations might be combined in bead model 

    def get_connected_location(self):
        if self.connection is None:
            return None
        else:
            return self.connection.other(self)

    def set_connection(self, connection, is_3prime_side_of_connection):
        self.connection = connection # TODO weakref? 
        self.is_3prime_side_of_connection = is_3prime_side_of_connection

    def get_nt_pos(self):
        try:
            pos = self.container.contour_to_nt_pos(self.address, round_nt=True)
        except:
            if self.address == 0:
                pos = 0
            elif self.address == 1:
                pos = self.container.num_nt-1
            else:
                raise
        return pos

    def __repr__(self):
        if self.on_fwd_strand:
            on_fwd = "on_fwd_strand"
        else:
            on_fwd = "on_rev_strand"
        return "<Location {}.{}[{:.2f},{:d}]>".format( self.container.name, self.type_, self.address, self.on_fwd_strand)
        
class Connection():
    """ Abstract base class for connection between two elements """
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
            raise Exception("OutOfBoundsError")

    def delete(self):
        self.A.container.connections.remove(self)
        if self.B.container is not self.A.container:
            self.B.container.connections.remove(self)
        self.A.connection = None
        self.B.connection = None

    def __repr__(self):
        return "<Connection {}--{}--{}]>".format( self.A, self.type_, self.B )
        

        
# class ConnectableElement(Transformable):
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

    def get_location_at(self, address, on_fwd_strand=True, new_type="crossover"):
        loc = None
        if (self.num_nt == 1):
            # import pdb
            # pdb.set_trace()
            ## Assumes that intrahelical connections have been made before crossovers
            for l in self.locations:
                if l.on_fwd_strand == on_fwd_strand and l.connection is None:
                    assert(loc is None)
                    loc = l
            # assert( loc is not None )
        else:
            for l in self.locations:
                if l.address == address and l.on_fwd_strand == on_fwd_strand:
                    assert(loc is None)
                    loc = l
        if loc is None:
            loc = Location( self, address=address, type_=new_type, on_fwd_strand=on_fwd_strand )
        return loc

    def get_connections_and_locations(self, connection_type=None, exclude=()):
        """ Returns a list with each entry of the form:
            connection, location_in_self, location_in_other """
        type_ = connection_type
        ret = []
        for c in self.connections:
            if (type_ is None or c.type_ == type_) and c.type_ not in exclude:
                if   c.A.container is self:
                    ret.append( [c, c.A, c.B] )
                elif c.B.container is self:
                    ret.append( [c, c.B, c.A] )
                else:
                    import pdb
                    pdb.set_trace()
                    raise Exception("Object contains connection that fails to refer to object")
        return ret

    def _connect(self, other, connection, in_3prime_direction=None):
        ## TODO fix circular references        
        A,B = [connection.A, connection.B]
        if in_3prime_direction is not None:
            A.is_3prime_side_of_connection = not in_3prime_direction
            B.is_3prime_side_of_connection = in_3prime_direction
            
        A.connection = B.connection = connection
        self.connections.append(connection)
        if other is not self:
            other.connections.append(connection)
        else:
            raise NotImplementedError("Segments cannot yet be connected to themselves; if you are attempting to make a circular object, try breaking the object into multiple segments")
        l = A.container.locations
        if A not in l: l.append(A)
        l = B.container.locations
        if B not in l: l.append(B)
        

    # def _find_connections(self, loc):
    #     return [c for c in self.connections if c.A == loc or c.B == loc]

class SegmentParticle(PointParticle):
    def __init__(self, type_, position, name="A", **kwargs):
        self.name = name
        self.contour_position = None
        PointParticle.__init__(self, type_, position, name=name, **kwargs)
        self.intrahelical_neighbors = []
        self.other_neighbors = []
        self.locations = []

    def get_intrahelical_above(self):
        """ Returns bead directly above self """
        # assert( len(self.intrahelical_neighbors) <= 2 )
        for b in self.intrahelical_neighbors:
            if b.get_contour_position(self.parent, self.contour_position) > self.contour_position:
                return b

    def get_intrahelical_below(self):
        """ Returns bead directly below self """
        # assert( len(self.intrahelical_neighbors) <= 2 )
        for b in self.intrahelical_neighbors:
            if b.get_contour_position(self.parent, self.contour_position) < self.contour_position:
                return b

    def _neighbor_should_be_added(self,b):
        if type(self.parent) != type(b.parent):
            return True

        c1 = self.contour_position
        c2 = b.get_contour_position(self.parent,c1)
        if c2 < c1:
            b0 = self.get_intrahelical_below()
        else:
            b0 = self.get_intrahelical_above()

        if b0 is not None:            
            c0 = b0.get_contour_position(self.parent,c1)
            if np.abs(c2-c1) < np.abs(c0-c1):
                ## remove b0
                self.intrahelical_neighbors.remove(b0)
                b0.intrahelical_neighbors.remove(self)
                return True
            else:
                return False
        return True
        
    def make_intrahelical_neighbor(self,b):
        add1 = self._neighbor_should_be_added(b)
        add2 = b._neighbor_should_be_added(self)
        if add1 and add2:
            # assert(len(b.intrahelical_neighbors) <= 1)
            # assert(len(self.intrahelical_neighbors) <= 1)
            self.intrahelical_neighbors.append(b)
            b.intrahelical_neighbors.append(self)

    def conceptual_get_position(self, context):
        """ 
        context: object

Q: does this function do too much?
Danger of mixing return values

Q: does context describe the system or present an argument?
        """

        ## Validate Inputs
        ...

        ## Algorithm
        """
context specifies:
  - kind of output: real space, nt within segment, fraction of segment
  - absolute or relative
  - constraints: e.g. if passing through
        """
        """
given context, provide the position
        input
"""

    def get_nt_position(self, seg, near_address=None):
        """ Returns the "address" of the nucleotide relative to seg in
        nucleotides, taking the shortest (intrahelical) contour length route to seg
        """
        if seg == self.parent:
            pos = self.contour_position
        else:
            pos = self.get_contour_position(seg,near_address)
        return seg.contour_to_nt_pos(pos)

    def get_contour_position(self,seg, address = None):
        """ TODO: fix paradigm where a bead maps to exactly one location in a polymer
        - One way: modify get_contour_position to take an optional argument that indicates where in the polymer you are looking from
        """

        if seg == self.parent:
            return self.contour_position
        else:
            cutoff = 30*3
            target_seg = seg

            ## depth-first search
            ## TODO cache distances to nearby locations?
            def descend_search_tree(seg, contour_in_seg, distance=0, visited_segs=None):
                nonlocal cutoff
                if visited_segs is None: visited_segs = []

                if seg == target_seg:
                    # pdb.set_trace()
                    ## Found a segment in our target
                    sign = 1 if contour_in_seg == 1 else -1
                    if sign == -1: assert( contour_in_seg == 0 )
                    if distance < cutoff: # TODO: check if this does anything
                        cutoff = distance
                    return [[distance, contour_in_seg+sign*seg.nt_pos_to_contour(distance)]], [(seg, contour_in_seg, distance)]
                if distance > cutoff:
                    return None,None
                    
                ret_list = []
                hist_list = []
                ## Find intrahelical locations in seg that we might pass through
                conn_locs = seg.get_connections_and_locations("intrahelical")
                if isinstance(target_seg, SingleStrandedSegment):
                    tmp = seg.get_connections_and_locations("sscrossover")
                    conn_locs = conn_locs + list(filter(lambda x: x[2].container == target_seg, tmp))
                for c,A,B in conn_locs:
                    if B.container in visited_segs: continue
                    dx = seg.contour_to_nt_pos( A.address, round_nt=False ) - seg.contour_to_nt_pos( contour_in_seg, round_nt=False)
                    dx = np.abs(dx)
                    results,history = descend_search_tree( B.container, B.address,
                                                   distance+dx, visited_segs + [seg] )
                    if results is not None:
                        ret_list.extend( results )
                        hist_list.extend( history )
                return ret_list,hist_list

            results,history = descend_search_tree(self.parent, self.contour_position)
            if results is None or len(results) == 0:
                raise Exception("Could not find location in segment") # TODO better error
            if address is not None:
                return sorted(results,key=lambda x:(x[0],(x[1]-address)**2))[0][1]
            else:
                return sorted(results,key=lambda x:x[0])[0][1]
            # nt_pos = self.get_nt_position(seg)
            # return seg.nt_pos_to_contour(nt_pos)

    def update_position(self, contour_position):
        self.contour_position = contour_position
        self.position = self.parent.contour_to_position(contour_position)
        if 'orientation_bead' in self.__dict__:
            o = self.orientation_bead
            o.contour_position = contour_position
            orientation = self.parent.contour_to_orientation(contour_position)
            if orientation is None:
                print("WARNING: local_twist is True, but orientation is None; using identity")
                orientation = np.eye(3)
            o.position = self.position + orientation.dot( np.array((Segment.orientation_bond.r0,0,0)) )
            
    def __repr__(self):
        return "<SegmentParticle {} on {}[{:.2f}]>".format( self.name, self.parent, self.contour_position)


## TODO break this class into smaller, better encapsulated pieces
class Segment(ConnectableElement, Group):

    """ Base class that describes a segment of DNA. When built from
    cadnano models, should not span helices """

    """Define basic particle types"""
    dsDNA_particle = ParticleType("D",
                                  diffusivity = 43.5,
                                  mass = 300,
                                  radius = 3,                 
                              )
    orientation_particle = ParticleType("O",
                                        diffusivity = 100,
                                        mass = 300,
                                        radius = 1,
                                    )

    # orientation_bond = HarmonicBond(10,2)
    orientation_bond = HarmonicBond(30,1.5, rRange = (0,500) )

    ssDNA_particle = ParticleType("S",
                                  diffusivity = 43.5,
                                  mass = 150,
                                  radius = 3,                 
                              )

    def __init__(self, name, num_nt, 
                 start_position = None,
                 end_position = None, 
                 segment_model = None,
                 **kwargs):

        if start_position is None: start_position = np.array((0,0,0))

        Group.__init__(self, name, children=[], **kwargs)
        ConnectableElement.__init__(self, connection_locations=[], connections=[])

        if 'segname' not in kwargs:
            self.segname = name
        # self.resname = name
        self.start_orientation = None
        self.twist_per_nt = 0

        self.beads = [c for c in self.children] # self.beads will not contain orientation beads

        self._bead_model_generation = 0    # TODO: remove?
        self.segment_model = segment_model # TODO: remove?

        self.strand_pieces = dict()
        for d in ('fwd','rev'):
            self.strand_pieces[d] = []

        self.num_nt = int(num_nt)
        if end_position is None:
            end_position = np.array((0,0,self.distance_per_nt*num_nt)) + start_position
        self.start_position = start_position
        self.end_position = end_position

        ## Used to assign cadnano names to beads
        self._generate_bead_callbacks = []
        self._generate_nucleotide_callbacks = []

        ## Set up interpolation for positions
        self._set_splines_from_ends()

        self.sequence = None

    def __repr__(self):
        return "<{} {}[{:d}]>".format( type(self), self.name, self.num_nt )

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
        return [self.contour_to_nt_pos(l.address) for l in self.locations]

    def insert_dna(self, at_nt: int, num_nt: int, seq=tuple()):
        assert(np.isclose(np.around(num_nt),num_nt))
        if at_nt < 0:
            raise ValueError("Attempted to insert DNA into {} at a negative location".format(self))
        if at_nt > self.num_nt-1:
            raise ValueError("Attempted to insert DNA into {} at beyond the end of the Segment".format(self))
        if num_nt < 0:
            raise ValueError("Attempted to insert DNA a negative amount of DNA into {}".format(self))

        num_nt = np.around(num_nt)
        nt_positions = self._get_location_positions()
        new_nt_positions = [p if p <= at_nt else p+num_nt for p in nt_positions]

        ## TODO: handle sequence

        self.num_nt = self.num_nt+num_nt

        for l,p in zip(self.locations, new_nt_positions):
            l.address = self.nt_pos_to_contour(p)

    def remove_dna(self, first_nt: int, last_nt: int):
        """ Removes nucleotides between first_nt and last_nt, inclusive """
        assert(np.isclose(np.around(first_nt),first_nt))
        assert(np.isclose(np.around(last_nt),last_nt))
        tmp = min((first_nt,last_nt))
        last_nt = max((first_nt,last_nt))
        fist_nt = tmp

        if first_nt < 0 or first_nt > self.num_nt-2:
            raise ValueError("Attempted to remove DNA from {} starting at an invalid location {}".format(self, first_nt))
        if last_nt < 1 or last_nt > self.num_nt-1:
            raise ValueError("Attempted to remove DNA from {} ending at an invalid location {}".format(self, last_nt))
        if first_nt == last_nt:
            return

        first_nt = np.around(first_nt)
        last_nt = np.around(last_nt)

        nt_positions = self._get_location_positions()

        bad_locations = list(filter(lambda p: p >= first_nt and p <= last_nt, nt_positions))
        if len(bad_locations) > 0:
            raise Exception("Attempted to remove DNA containing locations {} from {} between {} and {}".format(bad_locations,self,first_nt,last_nt))

        removed_nt = last_nt-first_nt+1
        new_nt_positions = [p if p <= last_nt else p-removed_nt for p in nt_positions]
        num_nt = self.num_nt-removed_nt

        if self.sequence is not None and len(self.sequence) == self.num_nt:
            self.sequence = [s for s,i in zip(self.sequence,range(self.num_nt)) 
                                if i < first_nt or i > last_nt]
            assert( len(self.sequence) == num_nt )

        self.num_nt = num_nt

        for l,p in zip(self.locations, new_nt_positions):
            l.address = self.nt_pos_to_contour(p)

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

        ids = Segment.__filter_contours(u, r, position_filter, contour_filter)
        if len(ids) == 0: return

        ## Translate
        r[ids,:] = r[ids,:] + dr[np.newaxis,:]
        self.set_splines(u,r)

    def rotate(self, rotation_matrix, about=None, position_filter=None, contour_filter=None):
        tck, u = self.position_spline_params
        r = self.contour_to_position(u)

        ids = Segment.__filter_contours(u, r, position_filter, contour_filter)
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
        u = np.linspace(0,1, max(3,self.num_nt//int(resolution)))
        s = u[:,np.newaxis]
        coords = (1-s)*r0 + s*r1
        self.set_splines(u, coords)

    def clear_all(self):
        Group.clear_all(self)  # TODO: use super?
        self.beads = []
        # for c,loc,other in self.get_connections_and_locations():
        #     loc.particle = None
        #     other.particle = None
        for l in self.locations:
            l.particle = None

    def contour_to_nt_pos(self, contour_pos, round_nt=False):
        nt = contour_pos*(self.num_nt) - 0.5
        if round_nt:
            assert( np.isclose(np.around(nt),nt) )
            nt = np.around(nt)
        return nt

    def nt_pos_to_contour(self,nt_pos):
        return (nt_pos+0.5)/(self.num_nt)

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

            orientation = rotationAboutAxis( axis, self.twist_per_nt*self.contour_to_nt_pos(s), normalizeAxis=False )
            orientation = orientation.dot(orientation0)
        else:
            q = interpolate.splev( s, self.quaternion_spline_params[0] )
            if len(q) > 1: q = np.array(q).T # TODO: is this needed?
            orientation = quaternion_to_matrix(q)

        return orientation

    def get_contour_sorted_connections_and_locations(self,type_):
        sort_fn = lambda c: c[1].address
        cl = self.get_connections_and_locations(type_)
        return sorted(cl, key=sort_fn)
    
    def randomize_unset_sequence(self):
        bases = list(seqComplement.keys())
        # bases = ['T']        ## FOR DEBUG
        if self.sequence is None:
            self.sequence = [random.choice(bases) for i in range(self.num_nt)]
        else:
            assert(len(self.sequence) == self.num_nt) # TODO move
            for i in range(len(self.sequence)):
                if self.sequence[i] is None:
                    self.sequence[i] = random.choice(bases)

    def _get_num_beads(self, max_basepairs_per_bead, max_nucleotides_per_bead ):
        raise NotImplementedError

    def _generate_one_bead(self, contour_position, nts):
        raise NotImplementedError

    def _generate_atomic_nucleotide(self, contour_position, is_fwd, seq, scale, strand_segment):
        """ Seq should include modifications like 5T, T3 Tsinglet; direction matters too """

        # print("Generating nucleotide at {}".format(contour_position))
        
        pos = self.contour_to_position(contour_position)
        orientation = self.contour_to_orientation(contour_position)

        """ deleteme
        ## TODO: move this code (?)
        if orientation is None:
            import pdb
            pdb.set_trace()
            axis = self.contour_to_tangent(contour_position)
            angleVec = np.array([1,0,0])
            if axis.dot(angleVec) > 0.9: angleVec = np.array([0,1,0])
            angleVec = angleVec - angleVec.dot(axis)*axis
            angleVec = angleVec/np.linalg.norm(angleVec)
            y = np.cross(axis,angleVec)
            orientation = np.array([angleVec,y,axis]).T
            ## TODO: improve placement of ssDNA
            # rot = rotationAboutAxis( axis, contour_position*self.twist_per_nt*self.num_nt, normalizeAxis=True )
            # orientation = rot.dot(orientation)
        else:
            orientation = orientation                            
        """
        key = seq
        nt_dict = canonicalNtFwd if is_fwd else canonicalNtRev

        atoms = nt_dict[ key ].generate() # TODO: clone?
        atoms.orientation = orientation.dot(atoms.orientation)
        if isinstance(self, SingleStrandedSegment):
            if scale is not None and scale != 1:
                for a in atoms:
                    a.position = scale*a.position
            atoms.position = pos - atoms.atoms_by_name["C1'"].collapsedPosition()
        else:
            if scale is not None and scale != 1:
                if atoms.sequence in ("A","G"):
                    r0 = atoms.atoms_by_name["N9"].position
                else:
                    r0 = atoms.atoms_by_name["N1"].position
                for a in atoms:
                    if a.name[-1] in ("'","P","T"):
                        a.position = scale*(a.position-r0) + r0
                    else:
                        a.fixed = 1
            atoms.position = pos
        
        atoms.contour_position = contour_position
        strand_segment.add(atoms)

        for callback in self._generate_nucleotide_callbacks:
            callback(atoms)

        return atoms

    def add_location(self, nt, type_, on_fwd_strand=True):
        ## Create location if needed, add to segment
        c = self.nt_pos_to_contour(nt)
        assert(c >= 0 and c <= 1)
        # TODO? loc = self.Location( address=c, type_=type_, on_fwd_strand=is_fwd )
        loc = Location( self, address=c, type_=type_, on_fwd_strand=on_fwd_strand )
        self.locations.append(loc)

    ## TODO? Replace with abstract strand-based model?

    def add_nick(self, nt, on_fwd_strand=True):
        self.add_3prime(nt,on_fwd_strand)
        self.add_5prime(nt+1,on_fwd_strand)

    def add_5prime(self, nt, on_fwd_strand=True):
        if isinstance(self,SingleStrandedSegment):
            on_fwd_strand = True
        self.add_location(nt,"5prime",on_fwd_strand)

    def add_3prime(self, nt, on_fwd_strand=True):
        if isinstance(self,SingleStrandedSegment):
            on_fwd_strand = True
        self.add_location(nt,"3prime",on_fwd_strand)

    def get_3prime_locations(self):
        return sorted(self.get_locations("3prime"),key=lambda x: x.address)
    
    def get_5prime_locations(self):
        ## TODO? ensure that data is consistent before _build_model calls
        return sorted(self.get_locations("5prime"),key=lambda x: x.address)

    def iterate_connections_and_locations(self, reverse=False):
        ## connections to other segments
        cl = self.get_contour_sorted_connections_and_locations()
        if reverse:
            cl = cl[::-1]
            
        for c in cl:
            yield c

    ## TODO rename
    def _add_strand_piece(self, strand_piece):
        """ Registers a strand segment within this object """

        ## TODO use weakref
        d = 'fwd' if strand_piece.is_fwd else 'rev'

        ## Validate strand_piece (ensure no clashes)
        for s in self.strand_pieces[d]:
            l,h = sorted((s.start,s.end))
            for value in (strand_piece.start,strand_piece.end):
                assert( value < l or value > h )

        ## Add strand_piece in correct order
        self.strand_pieces[d].append(strand_piece)
        self.strand_pieces[d] = sorted(self.strand_pieces[d],
                                       key = lambda x: x.start)

    ## TODO rename
    def get_strand_segment(self, nt_pos, is_fwd, move_at_least=0.5):
        """ Walks through locations, checking for crossovers """
        # if self.name in ("6-1","1-1"):
        #     import pdb
        #     pdb.set_trace()
        move_at_least = 0

        ## Iterate through locations
        # locations = sorted(self.locations, key=lambda l:(l.address,not l.on_fwd_strand), reverse=(not is_fwd))
        def loc_rank(l):
            nt = l.get_nt_pos()
            ## optionally add logic about type of connection
            return (nt, not l.on_fwd_strand)
        # locations = sorted(self.locations, key=lambda l:(l.address,not l.on_fwd_strand), reverse=(not is_fwd))
        locations = sorted(self.locations, key=loc_rank, reverse=(not is_fwd))
        # print(locations)

        for l in locations:
            # TODOTODO probably okay
            if l.address == 0:
                pos = 0.0
            elif l.address == 1:
                pos = self.num_nt-1
            else:
                pos = self.contour_to_nt_pos(l.address, round_nt=True)

            ## DEBUG


            ## Skip locations encountered before our strand
            # tol = 0.1
            # if is_fwd:
            #     if pos-nt_pos <= tol: continue 
            # elif   nt_pos-pos <= tol: continue
            if (pos-nt_pos)*(2*is_fwd-1) < move_at_least: continue
            ## TODO: remove move_at_least
            if np.isclose(pos,nt_pos):
                if l.is_3prime_side_of_connection: continue

            ## Stop if we found the 3prime end
            if l.on_fwd_strand == is_fwd and l.type_ == "3prime" and l.connection is None:
                # print("  found end at",l)
                return pos, None, None, None, None

            ## Check location connections
            c = l.connection
            if c is None: continue
            B = c.other(l)            

            ## Found a location on the same strand?
            if l.on_fwd_strand == is_fwd:
                # print("  passing through",l)
                # print("from {}, connection {} to {}".format(nt_pos,l,B))
                Bpos = B.get_nt_pos()
                return pos, B.container, Bpos, B.on_fwd_strand, 0.5
                
            ## Stop at other strand crossovers so basepairs line up
            elif c.type_ == "crossover":
                if nt_pos == pos: continue
                # print("  pausing at",l)
                return pos, l.container, pos+(2*is_fwd-1), is_fwd, 0

        raise Exception("Shouldn't be here")
        # print("Shouldn't be here")
        ## Made it to the end of the segment without finding a connection
        return 1*is_fwd, None, None, None

    def get_nearest_bead(self, contour_position):
        if len(self.beads) < 1: return None
        cs = np.array([b.contour_position for b in self.beads]) # TODO: cache
        # TODO: include beads in connections?
        i = np.argmin((cs - contour_position)**2)

        return self.beads[i]

    def _get_atomic_nucleotide(self, nucleotide_idx, is_fwd=True):
        d = 'fwd' if is_fwd else 'rev'
        for s in self.strand_pieces[d]:
            try:
                return s.get_nucleotide(nucleotide_idx)
            except:
                pass
        raise Exception("Could not find nucleotide in {} at {}.{}".format( self, nucleotide_idx, d ))

    def get_all_consecutive_beads(self, number):
        assert(number >= 1)
        ## Assume that consecutive beads in self.beads are bonded
        ret = []
        for i in range(len(self.beads)-number+1):
            tmp = [self.beads[i+j] for j in range(0,number)]
            ret.append( tmp )
        return ret   

    def _add_bead(self,b):
        
        # assert(b.parent is None)
        if b.parent is not None:
            b.parent.children.remove(b)
        self.add(b)
        self.beads.append(b) # don't add orientation bead
        if "orientation_bead" in b.__dict__: # TODO: think of a cleaner approach
            o = b.orientation_bead
            o.contour_position = b.contour_position
            if o.parent is not None:
                o.parent.children.remove(o)
            self.add(o)
            self.add_bond(b,o, Segment.orientation_bond, exclude=True)

    def _rebuild_children(self, new_children):
        # print("_rebuild_children on %s" % self.name)
        old_children = self.children
        old_beads = self.beads
        self.children = []
        self.beads = []

        if True:
            ## TODO: remove this if duplicates are never found 
            # print("Searching for duplicate particles...")
            ## Remove duplicates, preserving order
            tmp = []
            for c in new_children:
                if c not in tmp:
                    tmp.append(c)
                else:
                    print("  DUPLICATE PARTICLE FOUND!")
            new_children = tmp

        for b in new_children:
            self.beads.append(b)
            self.children.append(b)
            if "orientation_bead" in b.__dict__: # TODO: think of a cleaner approach
                self.children.append(b.orientation_bead)
            
        # tmp = [c for c in self.children if c not in old_children]
        # assert(len(tmp) == 0)
        # tmp = [c for c in old_children if c not in self.children]
        # assert(len(tmp) == 0)
        assert(len(old_children) == len(self.children))
        assert(len(old_beads) == len(self.beads))


    def _generate_beads(self, bead_model, max_basepairs_per_bead, max_nucleotides_per_bead):

        """ Generate beads (positions, types, etc) and bonds, angles, dihedrals, exclusions """
        ## TODO: decide whether to remove bead_model argument
        ##       (currently unused)

        ## First find points between-which beads must be generated
        # conn_locs = self.get_contour_sorted_connections_and_locations()
        # locs = [A for c,A,B in conn_locs]
        # existing_beads = [l.particle for l in locs if l.particle is not None]
        # if self.name == "S001":
        #     pdb.set_trace()

        # pdb.set_trace()
        existing_beads0 = { (l.particle, l.particle.get_contour_position(self,l.address))
                            for l in self.locations if l.particle is not None }
        existing_beads = sorted( list(existing_beads0), key=lambda bc: bc[1] )

        # if self.num_nt == 1 and all([l.particle is not None for l in self.locations]):
        #     pdb.set_trace()
        #     return

        for b,c in existing_beads:
            assert(b.parent is not None)

        ## Add ends if they don't exist yet
        ## TODOTODO: test 1 nt segments?
        if len(existing_beads) == 0 or existing_beads[0][0].get_nt_position(self,0) >= 0.5:
            # if len(existing_beads) > 0:            
            #     assert(existing_beads[0].get_nt_position(self) >= 0.5)
            b = self._generate_one_bead( self.nt_pos_to_contour(0), 0)
            existing_beads = (b,0) + existing_beads

        if existing_beads[-1][0].get_nt_position(self,1)-(self.num_nt-1) < -0.5 or len(existing_beads)==1:
            b = self._generate_one_bead( self.nt_pos_to_contour(self.num_nt-1), 0)
            existing_beads.append( (b,1) )
        assert(len(existing_beads) > 1)

        ## Walk through existing_beads, add beads between
        tmp_children = []       # build list of children in nice order
        last = None

        for I in range(len(existing_beads)-1):
            eb1,eb2 = [existing_beads[i][0] for i in (I,I+1)]
            ec1,ec2 = [existing_beads[i][1] for i in (I,I+1)]
            assert( (eb1,ec1) is not (eb2,ec2) )

            # if np.isclose(eb1.position[2], eb2.position[2]):
            #     import pdb
            #     pdb.set_trace()

            # print(" %s working on %d to %d" % (self.name, eb1.position[2], eb2.position[2]))
            e_ds = ec2-ec1
            num_beads = self._get_num_beads( e_ds, max_basepairs_per_bead, max_nucleotides_per_bead )

            ## Ensure there is a ssDNA bead between dsDNA beads
            if num_beads == 0 and isinstance(self,SingleStrandedSegment) and isinstance(eb1.parent,DoubleStrandedSegment) and isinstance(eb2.parent,DoubleStrandedSegment):
                num_beads = 1
            ## TODO similarly ensure there is a dsDNA bead between ssDNA beads

            ds = e_ds / (num_beads+1)
            nts = ds*self.num_nt
            eb1.num_nt += 0.5*nts
            eb2.num_nt += 0.5*nts

            ## Add beads
            if eb1.parent == self:
                tmp_children.append(eb1)

            s0 = ec1
            if last is not None:
                last.make_intrahelical_neighbor(eb1)
            last = eb1
            for j in range(num_beads):
                s = ds*(j+1) + s0
                # if self.name in ("51-2","51-3"):
                # if self.name in ("31-2",):
                #     print(" adding bead at {}".format(s))
                b = self._generate_one_bead(s,nts)

                last.make_intrahelical_neighbor(b)
                last = b
                tmp_children.append(b)

        last.make_intrahelical_neighbor(eb2)

        if eb2.parent == self:
            tmp_children.append(eb2)
        # if self.name in ("31-2",):
        #     pdb.set_trace()
        self._rebuild_children(tmp_children)

        for callback in self._generate_bead_callbacks:
            callback(self)

    def _regenerate_beads(self, max_nts_per_bead=4, ):
        ...
    

class DoubleStrandedSegment(Segment):

    """ Class that describes a segment of ssDNA. When built from
    cadnano models, should not span helices """

    def __init__(self, name, num_bp, start_position = np.array((0,0,0)),
                 end_position = None, 
                 segment_model = None,
                 local_twist = False,
                 num_turns = None,
                 start_orientation = None,
                 twist_persistence_length = 90,
                 **kwargs):
        
        self.helical_rise = 10.44
        self.distance_per_nt = 3.4
        Segment.__init__(self, name, num_bp,
                         start_position,
                         end_position, 
                         segment_model,
                         **kwargs)
        self.num_bp = self.num_nt

        self.local_twist = local_twist
        if num_turns is None:
            num_turns = float(num_bp) / self.helical_rise
        self.twist_per_nt = float(360 * num_turns) / num_bp

        if start_orientation is None:
            start_orientation = np.eye(3) # np.array(((1,0,0),(0,1,0),(0,0,1)))
        self.start_orientation = start_orientation
        self.twist_persistence_length = twist_persistence_length

        self.nicks = []

        self.start = self.start5 = Location( self, address=0, type_= "end5" )
        self.start3 = Location( self, address=0, type_ = "end3", on_fwd_strand=False )

        self.end = self.end3 = Location( self, address=1, type_ = "end3" )
        self.end5 = Location( self, address=1, type_= "end5", on_fwd_strand=False )
        # for l in (self.start5,self.start3,self.end3,self.end5):
        #     self.locations.append(l)

        ## TODO: initialize sensible spline for orientation

    ## Convenience methods
    ## TODO: add errors if unrealistic connections are made
    ## TODO: make connections automatically between unconnected strands
    def connect_start5(self, end3, type_="intrahelical", force_connection=False):
        if isinstance(end3, SingleStrandedSegment):
            end3 = end3.end3
        self._connect_ends( self.start5, end3, type_, force_connection = force_connection )
    def connect_start3(self, end5, type_="intrahelical", force_connection=False):
        if isinstance(end5, SingleStrandedSegment):
            end5 = end5.start5
        self._connect_ends( self.start3, end5, type_, force_connection = force_connection )
    def connect_end3(self, end5, type_="intrahelical", force_connection=False):
        if isinstance(end5, SingleStrandedSegment):
            end5 = end5.start5
        self._connect_ends( self.end3, end5, type_, force_connection = force_connection )
    def connect_end5(self, end3, type_="intrahelical", force_connection=False):
        if isinstance(end3, SingleStrandedSegment):
            end3 = end3.end3
        self._connect_ends( self.end5, end3, type_, force_connection = force_connection )

    def add_crossover(self, nt, other, other_nt, strands_fwd=(True,False), nt_on_5prime=True, type_="crossover"):
        """ Add a crossover between two helices """
        ## Validate other, nt, other_nt
        ##   TODO

        if isinstance(other,SingleStrandedSegment):
            other.add_crossover(other_nt, self, nt, strands_fwd[::-1], not nt_on_5prime)
        else:

            ## Create locations, connections and add to segments
            c = self.nt_pos_to_contour(nt)
            assert(c >= 0 and c <= 1)

            loc = self.get_location_at(c, strands_fwd[0])

            c = other.nt_pos_to_contour(other_nt)
            # TODOTODO: may need to subtract or add a little depending on 3prime/5prime
            assert(c >= 0 and c <= 1)
            other_loc = other.get_location_at(c, strands_fwd[1])
            self._connect(other, Connection( loc, other_loc, type_=type_ ))
            if nt_on_5prime:
                loc.is_3prime_side_of_connection = False
                other_loc.is_3prime_side_of_connection = True
            else:            
                loc.is_3prime_side_of_connection = True
                other_loc.is_3prime_side_of_connection = False

    ## Real work
    def _connect_ends(self, end1, end2, type_, force_connection):
        debug = False
        ## TODO remove self?
        ## validate the input
        for end in (end1, end2):
            assert( isinstance(end, Location) )
            assert( end.type_ in ("end3","end5") )
        assert( end1.type_ != end2.type_ )

        ## Remove other connections involving these points
        if end1.connection is not None:
            if debug: print("WARNING: reconnecting {}".format(end1))
            end1.connection.delete()
        if end2.connection is not None:
            if debug: print("WARNING: reconnecting {}".format(end2))
            end2.connection.delete()

        ## Create and add connection
        if end2.type_ == "end5":
            end1.container._connect( end2.container, Connection( end1, end2, type_=type_ ), in_3prime_direction=True )
        else:
            end2.container._connect( end1.container, Connection( end2, end1, type_=type_ ), in_3prime_direction=True )
    def _get_num_beads(self, contour, max_basepairs_per_bead, max_nucleotides_per_bead):
        # return int(contour*self.num_nt // max_basepairs_per_bead)
        return int(contour*(self.num_nt**2/(self.num_nt+1)) // max_basepairs_per_bead)

    def _generate_one_bead(self, contour_position, nts):
        pos = self.contour_to_position(contour_position)
        if self.local_twist:
            orientation = self.contour_to_orientation(contour_position)
            if orientation is None:
                print("WARNING: local_twist is True, but orientation is None; using identity")
                orientation = np.eye(3)
            opos = pos + orientation.dot( np.array((Segment.orientation_bond.r0,0,0)) )
            # if np.linalg.norm(pos) > 1e3:
            #     pdb.set_trace()
            assert(np.linalg.norm(opos-pos) < 10 )
            o = SegmentParticle( Segment.orientation_particle, opos, name="O",
                                 contour_position = contour_position,
                                 num_nt=nts, parent=self )
            bead = SegmentParticle( Segment.dsDNA_particle, pos, name="DNA",
                                    num_nt=nts, parent=self, 
                                    orientation_bead=o,
                                    contour_position=contour_position )

        else:
            bead = SegmentParticle( Segment.dsDNA_particle, pos, name="DNA",
                                    num_nt=nts, parent=self,
                                    contour_position=contour_position )
        self._add_bead(bead)
        return bead

class SingleStrandedSegment(Segment):

    """ Class that describes a segment of ssDNA. When built from
    cadnano models, should not span helices """

    def __init__(self, name, num_nt, start_position = None,
                 end_position = None, 
                 segment_model = None,
                 **kwargs):

        if start_position is None: start_position = np.array((0,0,0))
        self.distance_per_nt = 5
        Segment.__init__(self, name, num_nt, 
                         start_position,
                         end_position, 
                         segment_model,
                         **kwargs)

        self.start = self.start5 = Location( self, address=0, type_= "end5" ) # TODO change type_?
        self.end = self.end3 = Location( self, address=1, type_ = "end3" )
        # for l in (self.start5,self.end3):
        #     self.locations.append(l)

    def connect_end3(self, end5, force_connection=False):
        self._connect_end( end5,  _5_to_3 = True, force_connection = force_connection )

    def connect_start5(self, end3, force_connection=False):
        self._connect_end( end3,  _5_to_3 = False, force_connection = force_connection )

    def connect_5end(self, end3, force_connection=False): # TODO: change name or possibly deprecate
        print("WARNING: 'connect_5end' will be deprecated")
        return self.connect_start5( end3, force_connection=False)

    def _connect_end(self, other, _5_to_3, force_connection):
        assert( isinstance(other, Location) )
        if _5_to_3 == True:
            seg1 = self
            seg2 = other.container
            end1 = self.end3
            end2 = other
            assert(other.type_ != "end3")
            # if (other.type_ is not "end5"):
            #     print("WARNING: code does not prevent connecting 3prime to 3prime, etc")
        else:
            seg1 = other.container
            seg2 = self
            end1 = other
            end2 = self.start
            assert(other.type_ != "end5")
            # if (other.type_ is not "end3"):
            #     print("WARNING: code does not prevent connecting 3prime to 3prime, etc")

        ## Remove other connections involving these points
        if end1.connection is not None:
            print("WARNING: reconnecting {}".format(end1))
            end1.connection.delete()
        if end2.connection is not None:
            print("WARNING: reconnecting {}".format(end2))
            end2.connection.delete()

        conn = Connection( end1, end2, type_="intrahelical" )
        seg1._connect( seg2, conn, in_3prime_direction=True )


    def add_crossover(self, nt, other, other_nt, strands_fwd=(True,False), nt_on_5prime=True, type_='sscrossover'):
        """ Add a crossover between two helices """
        ## TODO Validate other, nt, other_nt

        assert(nt < self.num_nt)
        assert(other_nt < other.num_nt)
        if nt in (0,1,self.num_nt-1) and other_nt in (0,1,other.num_nt-1):
            if nt_on_5prime == True:
                other_end = other.start5 if strands_fwd[1] else other.end5
                self.connect_end3( other_end )
            else:
                other_end = other.end3 if strands_fwd[1] else other.start3
                self.connect_start5( other_end )
            return

        # c1 = self.nt_pos_to_contour(nt)
        # # TODOTODO
        # ## Ensure connections occur at ends, otherwise the structure doesn't make sense
        # # assert(np.isclose(c1,0) or np.isclose(c1,1))
        # assert(np.isclose(nt,0) or np.isclose(nt,self.num_nt-1))
        if nt == 0 and (self.num_nt > 1 or not nt_on_5prime):
            c1 = 0
        elif nt == self.num_nt-1:
            c1 = 1
        else:
            raise Exception("Crossovers can only be at the ends of an ssDNA segment")
        loc = self.get_location_at(c1, True)

        if other_nt == 0:
            c2 = 0
        elif other_nt == other.num_nt-1:
            c2 = 1
        else:
            c2 = other.nt_pos_to_contour(other_nt)

        if isinstance(other,SingleStrandedSegment):
            ## Ensure connections occur at opposing ends
            assert(np.isclose(other_nt,0) or np.isclose(other_nt,self.num_nt-1))
            other_loc = other.get_location_at( c2, True )
            # if ("22-2" in (self.name, other.name)):
            #     pdb.set_trace()
            if nt_on_5prime:
                self.connect_end3( other_loc )
            else:
                other.connect_end3( self )

        else:
            assert(c2 >= 0 and c2 <= 1)
            other_loc = other.get_location_at( c2, strands_fwd[1] )
            if nt_on_5prime:
                self._connect(other, Connection( loc, other_loc, type_="sscrossover" ), in_3prime_direction=True )
            else:
                other._connect(self, Connection( other_loc, loc, type_="sscrossover" ), in_3prime_direction=True )

    def _get_num_beads(self, contour, max_basepairs_per_bead, max_nucleotides_per_bead):
        return int(contour*(self.num_nt**2/(self.num_nt+1)) // max_basepairs_per_bead)
        # return int(contour*self.num_nt // max_nucleotides_per_bead)

    def _generate_one_bead(self, contour_position, nts):
        pos = self.contour_to_position(contour_position)
        b = SegmentParticle( Segment.ssDNA_particle, pos, 
                             name="NAS",
                             num_nt=nts, parent=self,
                             contour_position=contour_position )
        self._add_bead(b)
        return b
    
class StrandInSegment(Group):
    """ Represents a piece of an ssDNA strand within a segment """
    
    def __init__(self, segment, start, end, is_fwd):
        """ start/end should be provided expressed in nt coordinates, is_fwd tuples """
        Group.__init__(self)
        self.num_nt = 0
        # self.sequence = []
        self.segment = segment
        self.start = start
        self.end = end
        self.is_fwd = is_fwd

        nts = np.abs(end-start)+1
        self.num_nt = int(round(nts))
        assert( np.isclose(self.num_nt,nts) )
        segment._add_strand_piece(self)
    
    def _nucleotide_ids(self):
        nt0 = self.start # seg.contour_to_nt_pos(self.start)
        assert( np.abs(nt0 - round(nt0)) < 1e-5 )
        nt0 = int(round(nt0))
        assert( (self.end-self.start) >= 0 or not self.is_fwd )

        direction = (2*self.is_fwd-1)
        return range(nt0,nt0 + direction*self.num_nt, direction)

    def get_sequence(self):
        """ return 5-to-3 """
        # TODOTODO test
        seg = self.segment
        if self.is_fwd:
            return [seg.sequence[nt] for nt in self._nucleotide_ids()]
        else:
            return [seqComplement[seg.sequence[nt]] for nt in self._nucleotide_ids()]
    
    def get_contour_points(self):
        c0,c1 = [self.segment.nt_pos_to_contour(p) for p in (self.start,self.end)]
        return np.linspace(c0,c1,self.num_nt)

    def get_nucleotide(self, idx):
        """ idx expressed as nt coordinate within segment """

        lo,hi = sorted((self.start,self.end))
        if self.is_fwd:
            idx_in_strand = idx - lo
        else:
            idx_in_strand = hi - idx
        assert( np.isclose( idx_in_strand , int(round(idx_in_strand)) ) )
        assert(idx_in_strand >= 0)
        return self.children[int(round(idx_in_strand))]
    def __repr__(self):
        return "<StrandInSegment {}{}[{:.2f}-{:.2f},{:d}]>".format( self.parent.segname, self.segment.name, self.start, self.end, self.is_fwd)

            
class Strand(Group):
    """ Represents an entire ssDNA strand from 5' to 3' as it routes through segments """
    def __init__(self, segname = None, is_circular = False):
        Group.__init__(self)
        self.num_nt = 0
        self.children = self.strand_segments = []
        self.segname = segname
        self.is_circular = is_circular
        self.debug = False

    def __repr__(self):
        return "<Strand {}({})>".format( self.segname, self.num_nt )

    ## TODO disambiguate names of functions
    def add_dna(self, segment, start, end, is_fwd):
        """ start/end are given as nt """
        if np.abs(start-end) <= 0.9:
            if self.debug:
                print( "WARNING: segment constructed with a very small number of nts ({})".format(np.abs(start-end)) )
            # import pdb
            # pdb.set_trace()
        for s in self.strand_segments:
            if s.segment == segment and s.is_fwd == is_fwd:
                # assert( s.start not in (start,end) )
                # assert( s.end not in (start,end) )
                if s.start in (start,end) or s.end in (start,end):
                    raise CircularDnaError("Found circular DNA")

        s = StrandInSegment( segment, start, end, is_fwd )
        self.add( s )
        self.num_nt += s.num_nt

    def set_sequence(self,sequence): # , set_complement=True):
        ## validate input
        assert( len(sequence) >= self.num_nt )
        assert( np.all( [i in ('A','T','C','G') for i in sequence] ) )
        
        seq_idx = 0
        ## set sequence on each segment
        for s in self.children:
            seg = s.segment
            if seg.sequence is None:
                seg.sequence = [None for i in range(seg.num_nt)]

            if s.is_fwd:
                for nt in s._nucleotide_ids():
                    seg.sequence[nt] = sequence[seq_idx]
                    seq_idx += 1
            else:
                for nt in s._nucleotide_ids():
                    seg.sequence[nt] = seqComplement[sequence[seq_idx]]
                    seq_idx += 1

    # def get_sequence(self):
    #     sequence = []
    #     for ss in self.strand_segments:
    #         sequence.extend( ss.get_sequence() )

    #     assert( len(sequence) >= self.num_nt )
    #     ret = ["5"+sequence[0]] +\
    #           sequence[1:-1] +\
    #           [sequence[-1]+"3"]
    #     assert( len(ret) == self.num_nt )
    #     return ret

    def link_nucleotides(self, nt5, nt3):
        parent = nt5.parent if nt5.parent is nt3.parent else self
        o3,c3,c4,c2,h3 = [nt5.atoms_by_name[n]
                          for n in ("O3'","C3'","C4'","C2'","H3'")]
        p,o5,o1,o2,c5 = [nt3.atoms_by_name[n]
                         for n in ("P","O5'","O1P","O2P","C5'")]
        parent.add_bond( o3, p, None )
        parent.add_angle( c3, o3, p, None )
        for x in (o5,o1,o2):
            parent.add_angle( o3, p, x, None )
            parent.add_dihedral(c3, o3, p, x, None )
        for x in (c4,c2,h3):
            parent.add_dihedral(x, c3, o3, p, None )
        parent.add_dihedral(o3, p, o5, c5, None)

    def generate_atomic_model(self, scale, first_atomic_index):
        last = None
        resid = 1
        ## TODO relabel "strand_segment"
        strand_segment_count = 0
        for s in self.strand_segments:
            strand_segment_count += 1
            seg = s.segment
            contour = s.get_contour_points()
            # if s.end == s.start:
            #     pdb.set_trace()
            # assert(s.end != s.start)
            assert( s.num_nt == 1 or (np.linalg.norm( seg.contour_to_position(contour[-1]) - seg.contour_to_position(contour[0]) ) > 0.1) )
            nucleotide_count = 0
            for c,seq in zip(contour,s.get_sequence()):
                nucleotide_count += 1
                if last is None and not self.is_circular:
                    seq = "5"+seq
                if strand_segment_count == len(s.strand_segments) and nucleotide_count == s.num_nt and not self.is_circular:
                    seq = seq+"3"

                nt = seg._generate_atomic_nucleotide( c, s.is_fwd, seq, scale, s )

                ## Join last basepairs
                if last is not None:
                    self.link_nucleotides(last,nt)

                nt.__dict__['resid'] = resid
                resid += 1
                last = nt
                nt._first_atomic_index = first_atomic_index
                first_atomic_index += len(nt.children)

        if self.is_circular:
            self.link_nucleotides(last,self.strand_segments[0].children[0])

        return first_atomic_index

    def update_atomic_orientations(self,default_orientation):
        last = None
        resid = 1
        for s in self.strand_segments:
            seg = s.segment
            contour = s.get_contour_points()
            for c,seq,nt in zip(contour,s.get_sequence(),s.children):

                orientation = seg.contour_to_orientation(c)
                ## TODO: move this code (?)
                if orientation is None:
                    axis = seg.contour_to_tangent(c)
                    angleVec = np.array([1,0,0])
                    if axis.dot(angleVec) > 0.9: angleVec = np.array([0,1,0])
                    angleVec = angleVec - angleVec.dot(axis)*axis
                    angleVec = angleVec/np.linalg.norm(angleVec)
                    y = np.cross(axis,angleVec)
                    orientation = np.array([angleVec,y,axis]).T

                nt.orientation = orientation.dot(default_orientation) # this one should be correct

class SegmentModel(ArbdModel):
    def __init__(self, segments=[], local_twist=True, escapable_twist=True,
                 max_basepairs_per_bead=7,
                 max_nucleotides_per_bead=4,
                 dimensions=(1000,1000,1000), temperature=291,
                 timestep=50e-6, cutoff=50, 
                 decompPeriod=10000, pairlistDistance=None, 
                 nonbondedResolution=0,DEBUG=0):
        self.DEBUG = DEBUG
        if DEBUG > 0: print("Building ARBD Model")
        ArbdModel.__init__(self,segments,
                           dimensions, temperature, timestep, cutoff, 
                           decompPeriod, pairlistDistance=None,
                           nonbondedResolution=0)

        # self.max_basepairs_per_bead = max_basepairs_per_bead     # dsDNA
        # self.max_nucleotides_per_bead = max_nucleotides_per_bead # ssDNA
        self.children = self.segments = segments

        self._bonded_potential = dict() # cache for bonded potentials
        self._generate_strands()
        self.grid_potentials = []
        self._generate_bead_model( max_basepairs_per_bead, max_nucleotides_per_bead, local_twist, escapable_twist)

        self.useNonbondedScheme( nbDnaScheme )
        self.useTclForces = False

    def get_connections(self,type_=None,exclude=()):
        """ Find all connections in model, without double-counting """
        added=set()
        ret=[]
        for s in self.segments:
            items = [e for e in s.get_connections_and_locations(type_,exclude=exclude) if e[0] not in added]
            added.update([e[0] for e in items])
            ret.extend( list(sorted(items,key=lambda x: x[1].address)) )
        return ret
    
    def _recursively_get_beads_within_bonds(self,b1,bonds,done=()):
        ret = []
        done = list(done)
        done.append(b1)
        if bonds == 0:
            return [[]]

        for b2 in b1.intrahelical_neighbors:
            if b2 in done: continue
            for tmp in self._recursively_get_beads_within_bonds(b2, bonds-1, done):
                ret.append( [b2]+tmp )
        return ret

    def _get_intrahelical_beads(self,num=2):
        ## TODO: add check that this is not called before adding intrahelical_neighbors in _generate_bead_model

        assert(num >= 2)

        ret = []
        for s in self.segments:
            for b1 in s.beads:
                for bead_list in self._recursively_get_beads_within_bonds(b1, num-1):
                    assert(len(bead_list) == num-1)
                    if b1.idx < bead_list[-1].idx: # avoid double-counting
                        ret.append([b1]+bead_list)
        return ret


    def _get_intrahelical_angle_beads(self):
        return self._get_intrahelical_beads(num=3)

    def _get_potential(self, type_, kSpring, d, max_potential = None):
        key = (type_, kSpring, d, max_potential)
        if key not in self._bonded_potential:
            assert( kSpring >= 0 )
            if type_ == "bond":
                self._bonded_potential[key] = HarmonicBond(kSpring,d, rRange=(0,1200), max_potential=max_potential)
            elif type_ == "angle":
                self._bonded_potential[key] = HarmonicAngle(kSpring,d, max_potential=max_potential)
                # , resolution = 1, maxForce=0.1)
            elif type_ == "dihedral":
                self._bonded_potential[key] = HarmonicDihedral(kSpring,d, max_potential=max_potential)
            else:
                raise Exception("Unhandled potential type '%s'" % type_)
        return self._bonded_potential[key]
    def get_bond_potential(self, kSpring, d):
        assert( d > 0.2 )
        return self._get_potential("bond", kSpring, d)
    def get_angle_potential(self, kSpring, d):
        return self._get_potential("angle", kSpring, d)
    def get_dihedral_potential(self, kSpring, d, max_potential=None):
        while d > 180: d-=360
        while d < -180: d+=360
        return self._get_potential("dihedral", kSpring, d, max_potential)


    def _getParent(self, *beads ):
        if np.all( [b1.parent == b2.parent 
                    for b1,b2 in zip(beads[:-1],beads[1:])] ):
            return beads[0].parent
        else:
            return self

    def _get_twist_spring_constant(self, sep):
        """ sep in nt """
        kT = 0.58622522         # kcal/mol
        twist_persistence_length = 90  # set semi-arbitrarily as there is a large spread in literature
        ## <cos(q)> = exp(-s/Lp) = integrate( cos[x] exp(-A x^2), {x, 0, pi} ) / integrate( exp(-A x^2), {x, 0, pi} )
        ##   Assume A is small
        ## int[B_] :=  Normal[Integrate[ Series[Cos[x] Exp[-B x^2], {B, 0, 1}], {x, 0, \[Pi]}]/
        ##             Integrate[Series[Exp[-B x^2], {B, 0, 1}], {x, 0, \[Pi]}]]

        ## Actually, without assumptions I get fitFun below
        ## From http://www.annualreviews.org/doi/pdf/10.1146/annurev.bb.17.060188.001405
        ##   units "3e-19 erg cm/ 295 k K" "nm" =~ 73
        Lp = twist_persistence_length/0.34 

        fitFun = lambda x: np.real(erf( (4*np.pi*x + 1j)/(2*np.sqrt(x)) )) * np.exp(-1/(4*x)) / erf(2*np.sqrt(x)*np.pi) - np.exp(-sep/Lp)
        k = opt.leastsq( fitFun, x0=np.exp(-sep/Lp) )
        return k[0][0] * 2*kT*0.00030461742

    def extend(self, other, copy=True, include_strands=False):
        assert( isinstance(other, SegmentModel) )
        if copy:
            for s in other.segments:
                self.segments.append(deepcopy(s))
            if include_strands:
                for s in other.strands:
                    self.strands.append(deepcopy(s))
        else:
            for s in other.segments:
                self.segments.append(s)
            if include_strands:
                for s in other.strands:
                    self.strands.append(s)
        self._clear_beads()

    def update(self, segment , copy=False):
        assert( isinstance(segment, Segment) )
        if copy:
            segment = deepcopy(segment)
        self.segments.append(segment)
        self._clear_beads()

    """ Mapping between different resolution models """
    def clear_atomic(self):
        for strand in self.strands:
            for s in strand.children:
                s.clear_all()
        for seg in self.segments:
            for d in ('fwd','rev'):
                seg.strand_pieces[d] = []
        self._generate_strands()
        ## Clear sequence if needed
        for seg in self.segments:
            if seg.sequence is not None and len(seg.sequence) != seg.num_nt:
                seg.sequence = None

    def clear_beads(self):
        return self._clear_beads()

    def _clear_beads(self):
        ## TODO: deprecate
        for s in self.segments:
            try:
                s.clear_all()
            except:
                ...
        self.clear_all(keep_children=True)

        try:
            if len(self.strands[0].children[0].children) > 0:
                self.clear_atomic()
        except:
            ...

        ## Check that it worked
        assert( len([b for b in self]) == 0 )
        locParticles = []
        for s in self.segments:
            for c,A,B in s.get_connections_and_locations():
                for l in (A,B):
                    if l.particle is not None:
                        locParticles.append(l.particle)
        assert( len(locParticles) == 0 )
        assert( len([b for s in self.segments for b in s.beads]) == 0 )

    def _update_segment_positions(self, bead_coordinates):
        print("WARNING: called deprecated command '_update_segment_positions; use 'update_splines' instead")
        return self.update_splines(bead_coordinates)

    ## Operations on spline coordinates
    def translate(self, translation_vector, position_filter=None):
        for s in self.segments:
            s.translate(translation_vector, position_filter=position_filter)
        
    def rotate(self, rotation_matrix, about=None, position_filter=None):
        for s in self.segments:
            s.rotate(rotation_matrix, about=about, position_filter=position_filter)

    def get_center(self, include_ssdna=False):
        if include_ssdna:
            segments = self.segments
        else:
            segments = list(filter(lambda s: isinstance(s,DoubleStrandedSegment), 
                                   self.segments))
        centers = [s.get_center() for s in segments]
        weights = [s.num_nt*2 if isinstance(s,DoubleStrandedSegment) else s.num_nt for s in segments]
        # centers,weights = [np.array(a) for a in (centers,weights)]
        return np.average( centers, axis=0, weights=weights)

    def update_splines(self, bead_coordinates):
        """ Set new function for each segments functions
        contour_to_position and contour_to_orientation """

        for s in self.segments:
            # if s.name == "61-1":
            #     pdb.set_trace()

            cabs = s.get_connections_and_locations("intrahelical")
            if isinstance(s,SingleStrandedSegment):
                cabs = cabs + [[c,A,B] for c,A,B in s.get_connections_and_locations("sscrossover") if A.address == 0 or A.address == 1]
            if np.any( [B.particle is None for c,A,B in cabs] ):
                print( "WARNING: none type found in connection, skipping" )
                cabs = [e for e in cabs if e[2].particle is not None]

            def get_beads_and_contour_positions(s):
                ret_list = []
                def zip_bead_contour(beads,address=None):
                    if isinstance(address,list):
                        assert(False)
                        for b,a in zip(beads,address):
                            if b is None: continue
                            try:
                                ret_list.append((b, b.get_contour_position(s,a)))
                            except:
                                ...                                
                    else:
                        for b in beads:
                            if b is None: continue
                            try:
                                ret_list.append((b, b.get_contour_position(s,address)))
                            except:
                                ...
                    return ret_list
                
                ## Add beads from segment s
                beads_contours = zip_bead_contour(s.beads)
                beads_contours.extend( zip_bead_contour([A.particle for c,A,B in cabs]) )
                beads = set([b for b,c in beads_contours])

                ## Add nearby beads
                for c,A,B in cabs:
                    ## TODOTODO test?
                    filter_fn = lambda x: x is not None and x not in beads
                    bs = list( filter( filter_fn, B.particle.intrahelical_neighbors ) )
                    beads_contours.extend( zip_bead_contour( bs, A.address ) )
                    beads.update(bs)
                    for i in range(3):
                        bs = list( filter( filter_fn, [n for b in bs for n in b.intrahelical_neighbors] ) )
                        beads_contours.extend( zip_bead_contour( bs, A.address ) )
                        beads.update(bs)

                beads_contours = list(set(beads_contours))
                beads = list(beads)

                ## Skip beads that are None (some locations were not assigned a particle to avoid double-counting) 
                # beads = [b for b in beads if b is not None]
                assert( np.any([b is None for b,c in beads_contours]) == False )
                # beads = list(filter(lambda x: x[0] is not None, beads))

                if isinstance(s, DoubleStrandedSegment):
                    beads_contours = list(filter(lambda x: x[0].type_.name[0] == "D", beads_contours))
                return beads_contours

            beads_contours = get_beads_and_contour_positions(s)
            contours = [c for b,c in beads_contours]
            contours = np.array(contours, dtype=np.float16) # deliberately use low precision
            contours,ids1 = np.unique(contours, return_index=True)
            beads_contours = [beads_contours[i] for i in ids1]

            assert( np.any( (contours[:-1] - contours[1:])**2 >= 1e-8 ) )

            ## TODO: keep closest beads beyond +-1.5 if there are fewer than 2 beads
            tmp = []
            dist = 1
            while len(tmp) < 5 and dist < 3:
                tmp = list(filter(lambda bc: np.abs(bc[1]-0.5) < dist, beads_contours))
                dist += 0.1

            if len(tmp) <= 1:
                raise Exception("Failed to fit spline into segment {}".format(s))

            beads = [b for b,c in tmp]
            contours = [c for b,c in tmp]
            ids = [b.idx for b in beads]
            
            if len(beads) <= 1:
                pdb.set_trace()


            """ Get positions """
            positions = bead_coordinates[ids,:].T

            # print("BEADS NOT IN {}:".format(s))
            # for b,c in filter(lambda z: z[0] not in s.beads, zip(beads,contours)):
            #     print("  {}[{:03f}]: {:0.3f}".format(b.parent.name, b.contour_position, c) )


            tck, u = interpolate.splprep( positions, u=contours, s=0, k=1 )

            # if len(beads) < 8:
            #     ret = interpolate.splprep( positions, u=contours, s=0, k=1, full_output=1 )
            #     tck = ret[0][0]
            #     if ret[2] > 0:
            #         pdb.set_trace()

            # else:
            #     try:
            #         ret = interpolate.splprep( positions, u=contours, s=0, k=3, full_output=1 )
            #         tck = ret[0][0]
            #         if ret[2] > 0:
            #             pdb.set_trace()
            #     except:
            #         ret = interpolate.splprep( positions, u=contours, s=0, k=1, full_output=1 )
            #         tck = ret[0][0]
            #         if ret[2] > 0:
            #             pdb.set_trace()

            s.position_spline_params = (tck,u)

            """ Get orientation """
            def get_orientation_vector(bead,tangent):
                if 'orientation_bead' in bead.__dict__:
                    o = bead.orientation_bead
                    oVec = bead_coordinates[o.idx,:] - bead_coordinates[bead.idx,:]
                    oVec = oVec - oVec.dot(tangent)*tangent
                    oVec = oVec/np.linalg.norm(oVec)
                else:
                    oVec = None
                return oVec

            def remove_tangential_projection(vector, tangent):
                """ Assume tangent is normalized """
                v = vector - vector.dot(tangent)*tangent
                return v/np.linalg.norm(v)

            def get_orientation_vector(bead,tangent):
                if 'orientation_bead' in bead.__dict__:
                    o = bead.orientation_bead
                    oVec = bead_coordinates[o.idx,:] - bead_coordinates[bead.idx,:]
                    oVec = remove_tangential_projection(oVec,tangent)
                else:
                    oVec = None
                return oVec


            def get_previous_idx_if_none(list_):
                previous = None
                result = []
                i = 0
                for e in list_:
                    if e is None:
                        result.append(previous)
                    else:
                        previous = i
                    i+=1
                return result
            def get_next_idx_if_none(list_):
                tmp = get_previous_idx_if_none(list_[::-1])[::-1]
                return [ len(list_)-1-idx if idx is not None else idx for idx in tmp ]

            def fill_in_orientation_vectors(contours,orientation_vectors,tangents):
                result = []
                last_idx = get_previous_idx_if_none( orientation_vectors )
                next_idx = get_next_idx_if_none( orientation_vectors )
                none_idx = 0
                for c,ov,t in zip(contours,orientation_vectors,tangents):
                    if ov is not None:
                        result.append(ov)
                    else:
                        p = last_idx[none_idx]
                        n = next_idx[none_idx]
                        none_idx += 1
                        if p is None:
                            if n is None:
                                ## Should be quite rare; give something random if it happens
                                print("WARNING: unable to interpolate orientation")
                                o = np.array((1,0,0))
                                result.append( remove_tangential_projection(o,t) )
                            else:
                                o = orientation_vectors[n]
                                result.append( remove_tangential_projection(o,t) )
                        else:
                            if n is None:
                                o = orientation_vectors[p]
                                result.append( remove_tangential_projection(o,t) )
                            else:
                                cp,cn = [contours[i] for i in (p,n)]
                                op,on = [orientation_vectors[i] for i in (p,n)]
                                if (cn-cp) > 1e-6:
                                    o = ((cn-c)*op+(c-cp)*on)/(cn-cp)
                                else:
                                    o = op+on
                                result.append( remove_tangential_projection(o,t) )
                return result

            tangents = s.contour_to_tangent(contours)
            orientation_vectors = [get_orientation_vector(b,t) for b,t in zip(beads,tangents)]
            if len(beads) > 3 and any([e is not None for e in orientation_vectors] ):
                orientation_vectors = fill_in_orientation_vectors(contours, orientation_vectors, tangents)

                quats = []
                lastq = None
                for b,t,oVec in zip(beads,tangents,orientation_vectors):
                    y = np.cross(t,oVec)
                    assert( np.abs(np.linalg.norm(y) - 1) < 1e-2 )
                    q = quaternion_from_matrix( np.array([oVec,y,t]).T)
                    
                    if lastq is not None:
                        if q.dot(lastq) < 0:
                            q = -q
                    quats.append( q )
                    lastq = q

                # pdb.set_trace()
                quats = np.array(quats)
                # tck, u = interpolate.splprep( quats.T, u=contours, s=3, k=3 ) ;# cubic spline not as good
                tck, u = interpolate.splprep( quats.T, u=contours, s=0, k=1 )
                s.quaternion_spline_params = (tck,u)

    def _generate_bead_model(self,
                             max_basepairs_per_bead = 7,
                             max_nucleotides_per_bead = 4,
                             local_twist=False,
                             escapable_twist=True):
        ## TODO: deprecate
        self.generate_bead_model( max_basepairs_per_bead = max_basepairs_per_bead,
                                  max_nucleotides_per_bead = max_nucleotides_per_bead,
                                  local_twist=local_twist,
                                  escapable_twist=escapable_twist)

    def generate_bead_model(self,
                             max_basepairs_per_bead = 7,
                             max_nucleotides_per_bead = 4,
                             local_twist=False,
                             escapable_twist=True):

        self.children = self.segments # is this okay?
        self.clear_beads()

        segments = self.segments
        for s in segments:
            s.local_twist = local_twist

        """ Simplify connections """
        # d_nt = dict()           # 
        # for s in segments:
        #     d_nt[s] = 1.5/(s.num_nt-1)
        # for s in segments:
        #     ## replace consecutive crossovers with
        #     cl = sorted( s.get_connections_and_locations("crossover"), key=lambda x: x[1].address )
        #     last = None
        #     for entry in cl:
        #         c,A,B = entry
        #         if last is not None and \
        #            (A.address - last[1].address) < d_nt[s]:
        #             same_type = c.type_ == last[0].type_
        #             same_dest_seg = B.container == last[2].container
        #             if same_type and same_dest_seg:
        #                 if np.abs(B.address - last[2].address) < d_nt[B.container]:
        #                     ## combine
        #                     A.combine = last[1]
        #                     B.combine = last[2]

        #                     ...
        #         # if last is not None:
        #         #     s.bead_locations.append(last)
        #         ...
        #         last = entry
        # del d_nt

        """ Generate beads at intrahelical junctions """
        if self.DEBUG: print( "Adding intrahelical beads at junctions" )

        ## Loop through all connections, generating beads at appropriate locations
        for c,A,B in self.get_connections("intrahelical"):
            s1,s2 = [l.container for l in (A,B)]

            assert( A.particle is None )
            assert( B.particle is None )

            ## TODO: offload the work here to s1
            # TODOTODO
            a1,a2 = [l.address   for l in (A,B)]            
            for a in (a1,a2):
                assert( np.isclose(a,0) or np.isclose(a,1) )
            
            ## TODO improve this for combinations of ssDNA and dsDNA (maybe a1/a2 should be calculated differently)
            """ Search to see whether bead at location is already found """
            b = None
            if isinstance(s1,DoubleStrandedSegment):
                b = s1.get_nearest_bead(a1) 
                if b is not None:
                    assert( b.parent is s1 )
                    """ if above assertion is true, no problem here """
                    if np.abs(b.get_nt_position(s1) - s1.contour_to_nt_pos(a1)) > 0.5:
                        b = None
            if b is None and isinstance(s2,DoubleStrandedSegment):
                b = s2.get_nearest_bead(a2)
                if b is not None:
                    if np.abs(b.get_nt_position(s2) - s2.contour_to_nt_pos(a2)) > 0.5:
                        b = None


            if b is not None and b.parent not in (s1,s2):
                b = None

            if b is None:
                ## need to generate a bead
                if isinstance(s2,DoubleStrandedSegment):
                    b = s2._generate_one_bead(a2,0)
                else:
                    b = s1._generate_one_bead(a1,0)
            A.particle = B.particle = b
            b.is_intrahelical = True
            b.locations.extend([A,B])

        # pdb.set_trace()

        """ Generate beads at other junctions """
        for c,A,B in self.get_connections(exclude="intrahelical"):
            s1,s2 = [l.container for l in (A,B)]
            if A.particle is not None and B.particle is not None:
                continue
            # assert( A.particle is None )
            # assert( B.particle is None )

            ## TODO: offload the work here to s1/s2 (?)
            a1,a2 = [l.address   for l in (A,B)]

            def maybe_add_bead(location, seg, address, ):
                if location.particle is None:
                    b = seg.get_nearest_bead(address)
                    try:
                        distance = seg.contour_to_nt_pos(np.abs(b.contour_position-address))
                        max_distance =  min(max_basepairs_per_bead, max_nucleotides_per_bead)*0.5
                        if "is_intrahelical" in b.__dict__:
                            max_distance = 0.5
                        if distance >= max_distance:
                            raise Exception("except")

                        ## combine beads
                        b.update_position( 0.5*(b.contour_position + address) ) # avg position
                    except:
                        b = seg._generate_one_bead(address,0)
                    location.particle = b
                    b.locations.append(location)

            maybe_add_bead(A,s1,a1)
            maybe_add_bead(B,s2,a2)

        """ Some tests """
        for c,A,B in self.get_connections("intrahelical"):
            for l in (A,B):
                if l.particle is None: continue
                assert( l.particle.parent is not None )

        """ Generate beads in between """
        if self.DEBUG: print("Generating beads")
        for s in segments:
            s._generate_beads( self, max_basepairs_per_bead, max_nucleotides_per_bead )

        # """ Combine beads at junctions as needed """
        # for c,A,B in self.get_connections():
        #    ...

        # ## Debug
        # all_beads = [b for s in segments for b in s.beads]
        # positions = np.array([b.position for b in all_beads])
        # dists = positions[:,np.newaxis,:] - positions[np.newaxis,:,:] 
        # ids = np.where( np.sum(dists**2,axis=-1) + 0.02**2*np.eye(len(dists)) < 0.02**2 )
        # print( ids )
        # pdb.set_trace()

        """ Add intrahelical neighbors at connections """

        for c,A,B in self.get_connections("intrahelical"):
            b1,b2 = [l.particle for l in (A,B)]
            if b1 is b2:
                ## already handled by Segment._generate_beads
                pass
            else:
                for b in (b1,b2): assert( b is not None )
                b1.make_intrahelical_neighbor(b2)

        """ Reassign bead types """
        if self.DEBUG: print("Assigning bead types")
        beadtype_s = dict()
        beadtype_count = dict(D=0,O=0,S=0)

        def _assign_bead_type(bead, num_nt, decimals):
            num_nt0 = bead.num_nt
            bead.num_nt = np.around( np.float32(num_nt), decimals=decimals )
            char = bead.type_.name[0].upper()
            key = (char, bead.num_nt)
            if key in beadtype_s:
                bead.type_ = beadtype_s[key]
            else:
                t = deepcopy(bead.type_)
                t.__dict__["nts"] = bead.num_nt*2 if char in ("D","O") else bead.num_nt
                # t.name = t.name + "%03d" % (t.nts*10**decimals)
                t.name = char + "%03d" % (beadtype_count[char])
                t.mass = t.nts * 150
                t.diffusivity = 150 if t.nts == 0 else min( 50 / np.sqrt(t.nts/5), 150)
                beadtype_count[char] += 1
                if self.DEBUG: print( "{} --> {} ({})".format(num_nt0, bead.num_nt, t.name) )
                beadtype_s[key] = bead.type_ = t

        # (cluster_size[c-1])


        import scipy.cluster.hierarchy as hcluster
        beads = [b for s in segments for b in s if b.type_.name[0].upper() in ("D","O")]
        data = np.array([b.num_nt for b in beads])[:,np.newaxis]
        order = int(2-np.log10(2*max_basepairs_per_bead)//1)
        try:
            clusters = hcluster.fclusterdata(data, float(max_basepairs_per_bead)/500, criterion="distance")
            cluster_size = [np.mean(data[clusters == i]) for i in np.unique(clusters)]
        except:
            clusters = np.arange(len(data))+1
            cluster_size = data.flatten()
        for b,c in zip(beads,clusters):
            _assign_bead_type(b, cluster_size[c-1], decimals=order)

        beads = [b for s in segments for b in s if b.type_.name[0].upper() in ("S")]
        data = np.array([b.num_nt for b in beads])[:,np.newaxis]
        order = int(2-np.log10(max_nucleotides_per_bead)//1)
        try:
            clusters = hcluster.fclusterdata(data, float(max_nucleotides_per_bead)/500, criterion="distance")
            cluster_size = [np.mean(data[clusters == i]) for i in np.unique(clusters)]
        except:
            clusters = np.arange(len(data))+1
            cluster_size = data.flatten()
        for b,c in zip(beads,clusters):
            _assign_bead_type(b, cluster_size[c-1], decimals=order)

        self._apply_grid_potentials_to_beads(beadtype_s)

        # for bead in [b for s in segments for b in s]:
        #     num_nt0 = bead.num_nt
        #     # bead.num_nt = np.around( np.float32(num_nt), decimals=decimals )
        #     key = (bead.type_.name[0].upper(), bead.num_nt)
        #     if key in beadtype_s:
        #         bead.type_ = beadtype_s[key]
        #     else:
        #         t = deepcopy(bead.type_)
        #         t.__dict__["nts"] = bead.num_nt*2 if t.name[0].upper() in ("D","O") else bead.num_nt
        #         # t.name = t.name + "%03d" % (t.nts*10**decimals)
        #         t.name = t.name + "%.16f" % (t.nts)
        #         print( "{} --> {} ({})".format(num_nt0, bead.num_nt, t.name) )
        #         beadtype_s[key] = bead.type_ = t


        """ Update bead indices """
        self._countParticleTypes() # probably not needed here
        self._updateParticleOrder()

        """ Add intrahelical bond potentials """
        if self.DEBUG: print("Adding intrahelical bond potentials")
        dists = dict()          # intrahelical distances built for later use
        intra_beads = self._get_intrahelical_beads() 
        if self.DEBUG: print("  Adding %d bonds" % len(intra_beads))
        for b1,b2 in intra_beads:
            # assert( not np.isclose( np.linalg.norm(b1.collapsedPosition() - b2.collapsedPosition()), 0 ) )
            if np.linalg.norm(b1.collapsedPosition() - b2.collapsedPosition()) < 1:
                # print("WARNING: some beads are very close")
                ...

            parent = self._getParent(b1,b2)

            ## TODO: could be sligtly smarter about sep
            sep = 0.5*(b1.num_nt+b2.num_nt)

            conversion = 0.014393265 # units "pN/AA" kcal_mol/AA^2
            if b1.type_.name[0] == "D" and b2.type_.name[0] == "D":
                elastic_modulus_times_area = 1000 # pN http://markolab.bmbcb.northwestern.edu/marko/Cocco.CRP.02.pdf
                d = 3.4*sep
                k = conversion*elastic_modulus_times_area/d
            else:
                ## TODO: get better numbers our ssDNA model
                elastic_modulus_times_area = 800 # pN http://markolab.bmbcb.northwestern.edu/marko/Cocco.CRP.02.pdf
                d = 5*sep
                k = conversion*elastic_modulus_times_area/d
                # print(sep,d,k)
              
            if b1 not in dists:
                dists[b1] = dict()
            if b2 not in dists:
                dists[b2] = dict()
            # dists[b1].append([b2,sep])
            # dists[b2].append([b1,sep])
            dists[b1][b2] = sep
            dists[b2][b1] = sep

            if b1 is b2: continue

            # dists[[b1,b2]] = dists[[b2,b1]] = sep
            bond = self.get_bond_potential(k,d)
            parent.add_bond( b1, b2, bond, exclude=True )

        # for s in self.segments:
        #     sepsum = 0
        #     beadsum = 0
        #     for b1 in s.beads:
        #         beadsum += b1.num_nt
        #         for bead_list in self._recursively_get_beads_within_bonds(b1, 1):
        #             assert(len(bead_list) == 1)
        #             if b1.idx < bead_list[-1].idx: # avoid double-counting
        #                 for b2 in bead_list:
        #                     if b2.parent == b1.parent:
        #                         sepsum += dists[b1][b2]
        #     sepsum += sep
        #     print("Helix {}: bps {}, beads {}, separation {}".format(s.name, s.num_nt, beadsum, sepsum))

        """ Add intrahelical angle potentials """
        def get_effective_dsDNA_Lp(sep):

            """ The persistence length of our model was found to be a
            little off (probably due to NB interactions). This
            attempts to compensate """

            ## For 1 bp, Lp=559, for 25 Lp = 524
            beads_per_bp = sep/2
            Lp0 = 147
            # return 0.93457944*Lp0 ;# factor1
            return 0.97*Lp0 ;# factor2
            # factor = bead_per_bp * (0.954-0.8944
            # return Lp0 * bead_per_bp

        empirical_compensation_factor = max_basepairs_per_bead
        
        if self.DEBUG: print("Adding intrahelical angle potentials")
        for b1,b2,b3 in self._get_intrahelical_angle_beads():
            ## TODO: could be slightly smarter about sep
            sep = 0.5*(0.5*b1.num_nt+b2.num_nt+0.5*b3.num_nt)
            parent = self._getParent(b1,b2,b3)

            kT = 0.58622522         # kcal/mol
            if b1.type_.name[0] == "D" and b2.type_.name[0] == "D" and b3.type_.name[0] == "D":
                Lp = get_effective_dsDNA_Lp(sep)
                k = angle_spring_from_lp(sep,Lp)
                if local_twist:
                    k_dihed = 0.25*k
                    k *= 0.75    # reduce because orientation beads impose similar springs
                    dihed = self.get_dihedral_potential(k_dihed,180)
                    parent.add_dihedral(b1,b2,b2.orientation_bead,b3, dihed)


            else:
                ## TODO: get correct number from ssDNA model
                k = angle_spring_from_lp(sep,2)

            angle = self.get_angle_potential(k,180)
            parent.add_angle( b1, b2, b3, angle )

        """ Add intrahelical exclusions """
        if self.DEBUG: print("Adding intrahelical exclusions")
        beads = dists.keys()
        def _recursively_get_beads_within(b1,d,done=()):
            ret = []
            for b2,sep in dists[b1].items():
                if b2 in done: continue
                if sep < d:
                    ret.append( b2 )
                    done.append( b2 )
                    tmp = _recursively_get_beads_within(b2, d-sep, done)
                    if len(tmp) > 0: ret.extend(tmp)
            return ret

        exclusions = set()
        for b1 in beads:
            exclusions.update( [(b1,b) for b in _recursively_get_beads_within(b1, 20, done=[b1])] )

        if self.DEBUG: print("Adding %d exclusions" % len(exclusions))
        for b1,b2 in exclusions:
            parent = self._getParent(b1,b2)
            parent.add_exclusion( b1, b2 )

        """ Twist potentials """
        if local_twist:
            if self.DEBUG: print("Adding twist potentials")

            for b1 in beads:
                if "orientation_bead" not in b1.__dict__: continue
                for b2,sep in dists[b1].items():
                    if "orientation_bead" not in b2.__dict__: continue
                    if b2.idx < b1.idx: continue # Don't double-count

                    p1,p2 = [b.parent for b in (b1,b2)]
                    o1,o2 = [b.orientation_bead for b in (b1,b2)]

                    parent = self._getParent( b1, b2 )

                    """ Add heuristic 90 degree potential to keep orientation bead orthogonal """
                    Lp = get_effective_dsDNA_Lp(sep)
                    k = 0.5*angle_spring_from_lp(sep,Lp)
                    pot = self.get_angle_potential(k,90)
                    parent.add_angle(o1,b1,b2, pot)
                    parent.add_angle(b1,b2,o2, pot)

                    ## TODO: improve this
                    twist_per_nt = 0.5 * (p1.twist_per_nt + p2.twist_per_nt)
                    angle = sep*twist_per_nt
                    if angle > 360 or angle < -360:
                        print("WARNING: twist angle out of normal range... proceeding anyway")
                        # raise Exception("The twist between beads is too large")
                        
                    k = self._get_twist_spring_constant(sep)
                    if escapable_twist:
                        pot = self.get_dihedral_potential(k,angle,max_potential=1)
                    else:
                        pot = self.get_dihedral_potential(k,angle)
                    parent.add_dihedral(o1,b1,b2,o2, pot)

        def k_angle(sep):
            return  0.5*angle_spring_from_lp(sep,147)

        def k_xover_angle(sep):
            return 0.5 * k_angle(sep)

        def add_local_crossover_orientation_potential(b1,b2,is_parallel=True):
            u1,u2 = [b.get_intrahelical_above() for b in (b1,b2)]
            d1,d2 = [b.get_intrahelical_below() for b in (b1,b2)]

            k = k_xover_angle(sep=1)
            pot = self.get_angle_potential(k,120)

            if 'orientation_bead' in b1.__dict__:
                o = b1.orientation_bead
                self.add_angle( o,b1,b2, pot )
            if 'orientation_bead' in b2.__dict__:
                o = b2.orientation_bead
                self.add_angle( b1,b2,o, pot )

            if 'orientation_bead' in b1.__dict__:
                t0 = -90 if A.on_fwd_strand else 90
                if not is_parallel: t0 *= -1

                o1 = b1.orientation_bead
                if u2 is not None and isinstance(u2.parent,DoubleStrandedSegment):
                    k = k_xover_angle( dists[b2][u2] )
                    pot = self.get_dihedral_potential(k,t0)
                    self.add_dihedral( o1,b1,b2,u2, pot )
                elif d2 is not None and isinstance(d2.parent,DoubleStrandedSegment):
                    k = k_xover_angle( dists[b2][d2] )
                    pot = self.get_dihedral_potential(k,-t0)
                    self.add_dihedral( o1,b1,b2,d2, pot )
            if 'orientation_bead' in b2.__dict__:
                t0 = -90 if B.on_fwd_strand else 90
                if not is_parallel: t0 *= -1

                o2 = b2.orientation_bead
                if u1 is not None and isinstance(u1.parent,DoubleStrandedSegment):
                    k = k_xover_angle( dists[b1][u1] )
                    pot = self.get_dihedral_potential(k,t0)
                    self.add_dihedral( o2,b2,b1,u1, pot )
                elif d1 is not None and isinstance(d1.parent,DoubleStrandedSegment):
                    k = k_xover_angle( dists[b1][d1] )
                    pot = self.get_dihedral_potential(k,-t0)
                    self.add_dihedral( o2,b2,b1,d1, pot )


        """ Add connection potentials """
        for c,A,B in self.get_connections("terminal_crossover"):
            ## TODO: use a better description here
            b1,b2 = [loc.particle for loc in (c.A,c.B)]
            # pot = self.get_bond_potential(4,18.5)
            pot = self.get_bond_potential(4,12)
            self.add_bond(b1,b2, pot)

            dotProduct = b1.parent.contour_to_tangent(b1.contour_position).dot(
                b2.parent.contour_to_tangent(b2.contour_position) )

           ## Add potential to provide a particular orinetation
            if local_twist:
                add_local_crossover_orientation_potential(b1,b2, is_parallel=dotProduct > 0)


        """ Add connection potentials """
        for c,A,B in self.get_connections("sscrossover"):
            b1,b2 = [loc.particle for loc in (c.A,c.B)]
            ## TODO: improve parameters
            pot = self.get_bond_potential(4,6)
            self.add_bond(b1,b2, pot)

            ## Add potentials to provide a sensible orientation
            ## TODO refine potentials against all-atom simulation data
            """
            u1,u2 = [b.get_intrahelical_above() for b in (b1,b2)]
            d1,d2 = [b.get_intrahelical_below() for b in (b1,b2)]
            k_fn = k_angle
            if isinstance(b1.parent,DoubleStrandedSegment):
                a,b,c = (d1,b1,b2) if d1.parent is b1.parent else (u1,b1,b2)
            else:
                a,b,c = (d2,b2,b1) if d2.parent is b2.parent else (u2,b2,b1)
            """

            # pot = self.get_angle_potential( k_fn(dists[a][b]), 120 ) # a little taller than 90 degrees
            # self.add_angle( a,b,c, pot )
            # o = b.orientation_bead
            # angle=3
            # pot = self.get_angle_potential( 1, 0 ) # a little taller than 90 degrees
            # self.add_angle( a,b,c, pot )

            dotProduct = b1.parent.contour_to_tangent(b1.contour_position).dot(
                b2.parent.contour_to_tangent(b2.contour_position) )

            if local_twist:
                add_local_crossover_orientation_potential(b1,b2, is_parallel=dotProduct > 0)

        
        crossover_bead_pots = set()
        for c,A,B in self.get_connections("crossover"):
            b1,b2 = [loc.particle for loc in (A,B)]

            ## Avoid double-counting
            if (b1,b2,A.on_fwd_strand,B.on_fwd_strand) in crossover_bead_pots: continue
            crossover_bead_pots.add((b1,b2,A.on_fwd_strand,B.on_fwd_strand))
            crossover_bead_pots.add((b2,b1,B.on_fwd_strand,A.on_fwd_strand))
                
            pot = self.get_bond_potential(4,18.5)
            self.add_bond(b1,b2, pot)

            ## Get beads above and below
            u1,u2 = [b.get_intrahelical_above() for b in (b1,b2)]
            d1,d2 = [b.get_intrahelical_below() for b in (b1,b2)]
            dotProduct = b1.parent.contour_to_tangent(b1.contour_position).dot(
                b2.parent.contour_to_tangent(b2.contour_position) )
            if dotProduct < 0:
                tmp = d2
                d2  = u2
                u2  = tmp

            a = None
            if u1 is not None and u2 is not None:
                t0 = 0
                a,b,c,d = (u1,b1,b2,u2)
            elif d1 is not None and d2 is not None:
                t0 = 0
                a,b,c,d = (d1,b1,b2,d2 )
            elif d1 is not None and u2 is not None:
                t0 = 180
                a,b,c,d = (d1,b1,b2,u2)
            elif u1 is not None and d2 is not None:
                t0 = 180
                a,b,c,d = (u1,b1,b2,d2)

            if a is not None:
                k = k_xover_angle( dists[b][a]+dists[c][d] )
                pot = self.get_dihedral_potential(k,t0)
                self.add_dihedral( a,b,c,d,  pot )

            if local_twist:
                add_local_crossover_orientation_potential(b1,b2, is_parallel=dotProduct > 0)

            
        ## TODOTODO check that this works
        for crossovers in self.get_consecutive_crossovers():
            if local_twist: break
            ## filter crossovers
            for i in range(len(crossovers)-2):
                c1,A1,B1,dir1 = crossovers[i]
                c2,A2,B2,dir2 = crossovers[i+1]
                s1,s2 = [l.container for l in (A1,A2)]
                sep = A1.particle.get_nt_position(s1,near_address=A1.address) - A2.particle.get_nt_position(s2,near_address=A2.address)
                sep = np.abs(sep)

                assert(sep >= 0)

                n1,n2,n3,n4 = (B1.particle, A1.particle, A2.particle, B2.particle)

                """
                <cos(q)> = exp(-s/Lp) = integrate( cos[x] exp(-A x^2), {x, 0, pi} ) / integrate( exp(-A x^2), {x, 0, pi} )
                """

                ## From http://www.annualreviews.org/doi/pdf/10.1146/annurev.bb.17.060188.001405
                ##   units "3e-19 erg cm/ 295 k K" "nm" =~ 73
                Lp = s1.twist_persistence_length/0.34  # set semi-arbitrarily as there is a large spread in literature

                def get_spring(sep):
                    fitFun = lambda x: np.real(erf( (4*np.pi*x + 1j)/(2*np.sqrt(x)) )) * np.exp(-1/(4*x)) / erf(2*np.sqrt(x)*np.pi) - np.exp(-sep/Lp)
                    k = opt.leastsq( fitFun, x0=np.exp(-sep/Lp) )
                    return k[0][0] * 2*kT*0.00030461742

                k = get_spring( max(sep,2) )
                t0 = sep*s1.twist_per_nt # TODO weighted avg between s1 and s2
                
                # pdb.set_trace()
                if A1.on_fwd_strand: t0 -= 120
                if dir1 != dir2:
                    A2_on_fwd = not A2.on_fwd_strand
                else:
                    A2_on_fwd = A2.on_fwd_strand
                if A2_on_fwd: t0 += 120
   
                # t0 = (t0 % 360

                # if n2.idx == 0:
                #     print( n1.idx,n2.idx,n3.idx,n4.idx,k,t0,sep )
                if sep == 0 and n1 is not n4:
                    # pot = self.get_angle_potential(k,t0)
                    # self.add_angle( n1,n2,n4, pot )
                    pass
                else:
                    pot = self.get_dihedral_potential(k,t0)
                    self.add_dihedral( n1,n2,n3,n4, pot )

        # ## remove duplicate potentials; ## TODO ensure that they aren't added twice in the first place? 
        # self.remove_duplicate_terms()

    def walk_through_helices(segment, direction=1, processed_segments=None):
        """
    
        First and last segment should be same for circular helices
        """

        assert( direction in (1,-1) )
        if processed_segments == None:
            processed_segments = set()

        def segment_is_new_helix(s):
            return isinstance(s,DoubleStrandedSegment) and s not in processed_segments

        new_s = None
        s = segment
        ## iterate intrahelically connected dsDNA segments
        while segment_is_new_helix(s):
            conn_locs = s.get_contour_sorted_connections_and_locations("intrahelical")[::direction]
            processed_segments.add(new_s)        
            new_s = None
            new_dir = None
            for i in range(len(conn_locs)):
                c,A,B = conn_locs[i]
                ## TODO: handle change of direction
                # TODOTODO
                address = 1*(direction==-1)
                if A.address == address and segment_is_new_helix(B.container):
                    new_s = B.container
                    assert(B.address in (0,1))
                    new_dir = 2*(B.address == 0) - 1
                    break

            yield s,direction
            s = new_s   # will break if None
            direction = new_dir
            
        #     if new_s is None:
        #         break
        #     else:
        #         s = new_s
        # yield s
        ## return s


    def get_consecutive_crossovers(self):
        ## TODOTODO TEST
        crossovers = []
        processed_segments = set()
        for s1 in self.segments:
            if not isinstance(s1,DoubleStrandedSegment):
                continue
            if s1 in processed_segments: continue

            s0,d0 = list(SegmentModel.walk_through_helices(s1,direction=-1))[-1]

            # s,direction = get_start_of_helices()
            tmp = []
            for s,d in SegmentModel.walk_through_helices(s0,-d0):
                if s == s0 and len(tmp) > 0:
                    ## end of circular helix, only add first crossover
                    cl_list = s.get_contour_sorted_connections_and_locations("crossover")
                    if len(cl_list) > 0:
                        tmp.append( cl_list[::d][0] + [d] )
                else:
                    tmp.extend( [cl + [d] for cl in s.get_contour_sorted_connections_and_locations("crossover")[::d]] )
                processed_segments.add(s)
            crossovers.append(tmp)
        return crossovers

    def set_sequence(self, sequence, force=True):
        if force:
            self.strands[0].set_sequence(sequence)
        else:
            try:
                self.strands[0].set_sequence(sequence)
            except:
                ...
        for s in self.segments:
            s.randomize_unset_sequence()

    def _generate_strands(self):
        ## clear strands
        try:
            for s in self.strands:
                self.children.remove(s)
            for seg in self.segments:
                for d in ('fwd','rev'):
                    seg.strand_pieces[d] = []
        except:
            pass
        self.strands = strands = []

        """ Ensure unconnected ends have 5prime Location objects """
        for seg in self.segments:
            ## TODO move into Segment calls
            five_prime_locs = sum([seg.get_locations(s) for s in ("5prime","crossover","terminal_crossover")],[])
            three_prime_locs = sum([seg.get_locations(s) for s in ("3prime","crossover","terminal_crossover")],[])

            def is_start_5prime(l):
                return l.get_nt_pos() < 1 and l.on_fwd_strand
            def is_end_5prime(l):
                return l.get_nt_pos() > seg.num_nt-2 and not l.on_fwd_strand
            def is_start_3prime(l):
                return l.get_nt_pos() < 1 and not l.on_fwd_strand
            def is_end_3prime(l):
                return l.get_nt_pos() > seg.num_nt-2 and l.on_fwd_strand

            if seg.start5.connection is None:
                if len(list(filter( is_start_5prime, five_prime_locs ))) == 0:
                    seg.add_5prime(0) # TODO ensure this is the same place

            if 'end5' in seg.__dict__ and seg.end5.connection is None:
                if len(list(filter( is_end_5prime, five_prime_locs ))) == 0:
                    seg.add_5prime(seg.num_nt-1,on_fwd_strand=False)

            if 'start3' in seg.__dict__ and seg.start3.connection is None:
                if len(list(filter( is_start_3prime, three_prime_locs ))) == 0:
                    seg.add_3prime(0,on_fwd_strand=False)

            if seg.end3.connection is None:
                if len(list(filter( is_end_3prime, three_prime_locs ))) == 0:
                    seg.add_3prime(seg.num_nt-1)

            # print( [(l,l.get_connected_location()) for l in seg.locations] )
            # addresses = np.array([l.address for l in seg.get_locations("5prime")])
            # if not np.any( addresses == 0 ):
            #     ## check if end is connected
        # for c,l,B in self.get_connections_and_locations():
        #     if c[0]

        """ Build strands from connectivity of helices """
        def _recursively_build_strand(strand, segment, pos, is_fwd, mycounter=0, move_at_least=0.5):
            seg = segment
            history = []
            while True:
                mycounter+=1
                if mycounter > 10000:
                    raise Exception("Too many iterations")

                #if seg.name == "22-1" and pos > 140:
                # if seg.name == "22-2":
                #     import pdb
                #     pdb.set_trace()

                end_pos, next_seg, next_pos, next_dir, move_at_least = seg.get_strand_segment(pos, is_fwd, move_at_least)
                history.append((seg,pos,end_pos,is_fwd))
                try:
                    strand.add_dna(seg, pos, end_pos, is_fwd)
                except CircularDnaError:
                    ## Circular DNA was found
                    break
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    # import pdb
                    # pdb.set_trace()
                    # seg.get_strand_segment(pos, is_fwd, move_at_least)
                    # strand.add_dna(seg, pos, end_pos, is_fwd)
                    raise

                if next_seg is None:
                    break
                else:
                    seg,pos,is_fwd = (next_seg, next_pos, next_dir)
            strand.history = list(history)
            return history

        strand_counter = 0
        history = []
        for seg in self.segments:
            locs = filter(lambda l: l.connection is None, seg.get_5prime_locations())
            if locs is None: continue
            # for pos, is_fwd in locs:
            for l in locs:
                # print("Tracing",l)
                # TODOTODO
                pos = seg.contour_to_nt_pos(l.address, round_nt=True)
                is_fwd = l.on_fwd_strand
                s = Strand(segname="S{:03d}".format(len(strands)))
                strand_history = _recursively_build_strand(s, seg, pos, is_fwd)
                history.append((l,strand_history))
                # print("{} {}".format(seg.name,s.num_nt))
                strands.append(s)

        ## Trace circular DNA
        def strands_cover_segment(segment, is_fwd=True):
            direction = 'fwd' if is_fwd else 'rev'
            nt = 0
            for sp in segment.strand_pieces[direction]:
                nt += sp.num_nt
            return nt == segment.num_nt

        def find_nt_not_in_strand(segment, is_fwd=True):
            fwd_str = 'fwd' if is_fwd else 'rev'

            def check(val):
                assert(val >= 0 and val < segment.num_nt)
                # print("find_nt_not_in_strand({},{}) returning {}".format(
                #     segment, is_fwd, val))
                return val

            if is_fwd:
                last = -1
                for sp in segment.strand_pieces[fwd_str]:
                    if sp.start-last > 1:
                        return check(last+1)
                    last = sp.end
                return check(last+1)
            else:
                last = segment.num_nt
                for sp in segment.strand_pieces[fwd_str]:
                    if last-sp.end > 1:
                        return check(last-1)
                    last = sp.start
                return check(last-1)

        def add_strand_if_needed(seg,is_fwd):
            history = []
            if not strands_cover_segment(seg, is_fwd):
                pos = nt = find_nt_not_in_strand(seg, is_fwd)
                s = Strand(is_circular = True)
                history = _recursively_build_strand(s, seg, pos, is_fwd)
                strands.append(s)
            return history

        for seg in self.segments:
            add_strand_if_needed(seg,True)
            if isinstance(seg, DoubleStrandedSegment):
                add_strand_if_needed(seg,False)

        self.strands = sorted(strands, key=lambda s:s.num_nt)[::-1]
        def check_strands():
            dsdna = filter(lambda s: isinstance(s,DoubleStrandedSegment), self.segments)
            for s in dsdna:
                nt_fwd = nt_rev = 0
                for sp in s.strand_pieces['fwd']:
                    nt_fwd += sp.num_nt
                for sp in s.strand_pieces['rev']:
                    nt_rev += sp.num_nt
                assert( nt_fwd == s.num_nt and nt_rev == s.num_nt )
                # print("{}: {},{} (fwd,rev)".format(s.name, nt_fwd/s.num_nt,nt_rev/s.num_nt))
        check_strands()

        ## relabel segname
        counter = 0
        for s in self.strands:
            if s.segname is None:
                s.segname = "D%03d" % counter
            counter += 1

    def _assign_basepairs(self):
        ## Assign basepairs
        for seg in self.segments:
            if isinstance(seg, DoubleStrandedSegment):
                strands1 = seg.strand_pieces['fwd'] # already sorted
                strands2 = seg.strand_pieces['rev']

                nts1 = [nt for s in strands1 for nt in s.children]
                nts2 = [nt for s in strands2 for nt in s.children[::-1]]
                assert(len(nts1) == len(nts2))
                for nt1,nt2 in zip(nts1,nts2):
                    ## TODO weakref
                    nt1.basepair = nt2
                    nt2.basepair = nt1

    def _update_orientations(self,orientation):
        for s in self.strands:
            s.update_atomic_orientations(orientation)

    def write_atomic_ENM(self, output_name, lattice_type=None):
        ## TODO: ensure atomic model was generated already
        if lattice_type is None:
            try:
                lattice_type = self.lattice_type
            except:
                lattice_type = "square"
        else:
            try:
                if lattice_type != self.lattice_type:
                    print("WARNING: printing ENM with a lattice type ({}) that differs from model's lattice type ({})".format(lattice_type,self.lattice_type))
            except:
                pass

        if lattice_type == "square":
            enmTemplate = enmTemplateSQ
        elif lattice_type == "honeycomb":
            enmTemplate = enmTemplateHC
        else:
            raise Exception("Lattice type '%s' not supported" % self.latticeType)

        ## TODO: allow ENM to be created without first building atomic model
        noStackPrime = 0
        noBasepair = 0
        with open("%s.exb" % output_name,'w') as fh:
            # natoms=0

            for seg in self.segments:
                ## Continue unless dsDNA
                if not isinstance(seg,DoubleStrandedSegment): continue
                for strand_piece in seg.strand_pieces['fwd'] + seg.strand_pieces['rev']:
                    for nt1 in strand_piece.children:
                        other = []
                        nt2 = nt1.basepair
                        if strand_piece.is_fwd:
                            other.append((nt2,'pair'))

                        nt2 = nt2.get_intrahelical_above()
                        if nt2 is not None and strand_piece.is_fwd:
                            ## TODO: check if this already exists
                            other.append((nt2,'paircross'))

                        nt2 = nt1.get_intrahelical_above()
                        if nt2 is not None:
                            other.append((nt2,'stack'))
                            nt2 = nt2.basepair
                            if nt2 is not None and strand_piece.is_fwd:
                                other.append((nt2,'cross'))

                        for nt2,key in other:
                            """
                            if np.linalg.norm(nt2.position-nt1.position) > 7:
                                import pdb
                                pdb.set_trace()
                            """
                            key = ','.join((key,nt1.sequence[0],nt2.sequence[0]))
                            for n1, n2, d in enmTemplate[key]:
                                d = float(d)
                                k = 0.1
                                if lattice_type == 'honeycomb':
                                    correctionKey = ','.join((key,n1,n2))
                                    assert(correctionKey in enmCorrectionsHC)
                                    dk,dr = enmCorrectionsHC[correctionKey]
                                    k  = float(dk)
                                    d += float(dr)

                                i = nt1._get_atomic_index(name=n1)
                                j = nt2._get_atomic_index(name=n2)
                                fh.write("bond %d %d %f %.2f\n" % (i,j,k,d))

            # print("NO STACKS found for:", noStackPrime)
            # print("NO BASEPAIRS found for:", noBasepair)

        ## Loop dsDNA regions
        push_bonds = []
        processed_segs = set()
        ## TODO possibly merge some of this code with SegmentModel.get_consecutive_crossovers()
        for segI in self.segments: # TODOTODO: generalize through some abstract intrahelical interface that effectively joins "segments", for now interhelical bonds that cross intrahelically-connected segments are ignored
            if not isinstance(segI,DoubleStrandedSegment): continue

            ## Loop over dsDNA regions connected by crossovers
            conn_locs = segI.get_contour_sorted_connections_and_locations("crossover")
            other_segs = list(set([B.container for c,A,B in conn_locs]))
            for segJ in other_segs:
                if (segI,segJ) in processed_segs:
                    continue
                processed_segs.add((segI,segJ))
                processed_segs.add((segJ,segI))

                ## TODO perhaps handle ends that are not between crossovers

                ## Loop over ordered pairs of crossovers between the two
                cls = filter(lambda x: x[-1].container == segJ, conn_locs)
                cls = sorted( cls, key=lambda x: x[1].get_nt_pos() )
                for cl1,cl2 in zip(cls[:-1],cls[1:]):
                    c1,A1,B1 = cl1
                    c2,A2,B2 = cl2

                    ntsI1,ntsI2 = [segI.contour_to_nt_pos(A.address) for A in (A1,A2)]
                    ntsJ1,ntsJ2 = [segJ.contour_to_nt_pos(B.address) for B in (B1,B2)]
                    ntsI = ntsI2-ntsI1+1
                    ntsJ = ntsJ2-ntsJ1+1
                    assert( np.isclose( ntsI, int(round(ntsI)) ) )
                    assert( np.isclose( ntsJ, int(round(ntsJ)) ) )
                    ntsI,ntsJ = [int(round(i)) for i in (ntsI,ntsJ)]

                    ## Find if dsDNA "segments" are pointing in same direction
                    ## could move this block out of the loop
                    tangentA = segI.contour_to_tangent(A1.address)
                    tangentB = segJ.contour_to_tangent(B1.address)
                    dot1 = tangentA.dot(tangentB)
                    tangentA = segI.contour_to_tangent(A2.address)
                    tangentB = segJ.contour_to_tangent(B2.address)
                    dot2 = tangentA.dot(tangentB)

                    if dot1 > 0.5 and dot2 > 0.5:
                        ...
                    elif dot1 < -0.5 and dot2 < -0.5:
                        ## TODO, reverse
                        ...
                        # print("Warning: {} and {} are on antiparallel helices (not yet implemented)... skipping".format(A1,B1))
                        continue
                    else:
                        # print("Warning: {} and {} are on helices that do not point in similar direction... skipping".format(A1,B1))
                        continue

                    ## Go through each nucleotide between the two
                    for ijmin in range(min(ntsI,ntsJ)):
                        i=j=ijmin
                        if ntsI < ntsJ:
                            j = int(round(float(ntsJ*i)/ntsI))
                        elif ntsJ < ntsI:
                            i = int(round(float(ntsI*j)/ntsJ))
                        ntI_idx = int(round(ntsI1+i))
                        ntJ_idx = int(round(ntsJ1+j))

                        ## Skip nucleotides that are too close to crossovers
                        if i < 11 or j < 11: continue
                        if ntsI2-ntI_idx < 11 or ntsJ2-ntJ_idx < 11: continue

                        ## Find phosphates at ntI/ntJ
                        for direction in [True,False]:
                            try:
                                i = segI._get_atomic_nucleotide(ntI_idx, direction)._get_atomic_index(name="P")
                                j = segJ._get_atomic_nucleotide(ntJ_idx, direction)._get_atomic_index(name="P")
                                push_bonds.append((i,j))
                            except:
                                # print("WARNING: could not find 'P' atom in {}:{} or {}:{}".format( segI, ntI_idx, segJ, ntJ_idx ))
                                ...

        # print("PUSH BONDS:", len(push_bonds))

        if not self.useTclForces:
            with open("%s.exb" % output_name, 'a') as fh:
                for i,j in push_bonds:
                    fh.write("bond %d %d %f %.2f\n" % (i,j,1.0,31))
        else:
            flat_push_bonds = list(sum(push_bonds))
            atomList = list(set( flat_push_bonds ))
            with open("%s.forces.tcl" % output_name,'w') as fh:
                fh.write("set atomList {%s}\n\n" %
                         " ".join([str(x-1) for x in  atomList]) )
                fh.write("set bonds {%s}\n" %
                         " ".join([str(x-1) for x in flat_push_bonds]) )
                fh.write("""
foreach atom $atomList {
    addatom $atom
}

proc calcforces {} {
    global atomList bonds
    loadcoords rv

    foreach i $atomList {
        set force($i) {0 0 0}
    }

    foreach {i j} $bonds {
        set rvec [vecsub $rv($j) $rv($i)]
        # lassign $rvec x y z
        # set r [expr {sqrt($x*$x+$y*$y+$z*$z)}]
        set r [getbond $rv($j) $rv($i)]
        set f [expr {2*($r-31.0)/$r}]
        vecadd $force($i) [vecscale $f $rvec]
        vecadd $force($j) [vecscale [expr {-1.0*$f}] $rvec]
    }

    foreach i $atomList {
        addforce $i $force($i)
    }

}
""")

    def dimensions_from_structure( self, padding_factor=1.5, isotropic=False ):
        positions = []
        for s in self.segments:
            positions.append(s.contour_to_position(0))
            positions.append(s.contour_to_position(0.5))
            positions.append(s.contour_to_position(1))
        positions = np.array(positions)
        dx,dy,dz = [(np.max(positions[:,i])-np.min(positions[:,i])+30)*padding_factor for i in range(3)]
        if isotropic:
            dx = dy = dz = max((dx,dy,dz))
        return [dx,dy,dz]

    def add_grid_potential(self, grid_file, scale=1, per_nucleotide=True):
        grid_file = Path(grid_file)
        if not grid_file.is_file():
            raise ValueError("Grid file {} does not exist".format(grid_file))
        if not grid_file.is_absolute():
            grid_file = Path.cwd() / grid_file
        self.grid_potentials.append((grid_file,scale,per_nucleotide))

    def _apply_grid_potentials_to_beads(self, bead_type_dict):
        if len(self.grid_potentials) > 1:
            raise NotImplementedError("Multiple grid potentials are not yet supported")

        for grid_file, scale, per_nucleotide in self.grid_potentials:
            for key,particle_type in bead_type_dict.items():
                if particle_type.name[0] == "O": continue
                s = scale*particle_type.nts if per_nucleotide else scale
                try:
                    particle_type.grid = particle_type.grid + (grid_file, s)
                except:
                    particle_type.grid = tuple((grid_file, s))

    def _generate_atomic_model(self, scale=1):
        ## TODO: deprecate
        self.generate_atomic_model(scale=scale)

    def generate_atomic_model(self, scale=1):
        self.clear_beads()
        self.children = self.strands # TODO: is this going to be okay? probably
        first_atomic_index = 0
        for s in self.strands:
            first_atomic_index = s.generate_atomic_model(scale,first_atomic_index)
        self._assign_basepairs()
        return

        ## Angle optimization
        angles = np.linspace(-180,180,180)
        score = []
        for a in angles:
            o = rotationAboutAxis([0,0,1], a)
            sum2 = count = 0
            for s in self.strands:
                s.update_atomic_orientations(o)
                for s1,s2 in zip(s.strand_segments[:-1],s.strand_segments[1:]):
                    nt1 = s1.children[-1]
                    nt2 = s2.children[0]
                    o3 = nt1.atoms_by_name["O3'"]
                    p = nt2.atoms_by_name["P"]
                    sum2 += np.sum((p.collapsedPosition()-o3.collapsedPosition())**2)
                    count += 1
            score.append(sum2/count)
        print(angles[np.argmin(score)])
        print(score)

    def vmd_tube_tcl(self, file_name="drawTubes.tcl"):
        with open(file_name, 'w') as tclFile:
            tclFile.write("## beginning TCL script \n")

            def draw_tube(segment,radius_value=10, color="cyan", resolution=5):
                tclFile.write("## Tube being drawn... \n")
                
                contours = np.linspace(0,1, max(2,1+segment.num_nt//resolution) )
                rs = [segment.contour_to_position(c) for c in contours]
               
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
            
            ## iterate through the model segments
            for s in self.segments:
                if isinstance(s,DoubleStrandedSegment):
                    tclFile.write("## dsDNA! \n")
                    draw_tube(s,10,"cyan")
                elif isinstance(s,SingleStrandedSegment):
                    tclFile.write("## ssDNA! \n")
                    draw_tube(s,3,"orange",resolution=1.5)
                else:
                    raise Exception ("your model includes beads that are neither ssDNA nor dsDNA")
            ## tclFile complete
            tclFile.close()

    def vmd_cylinder_tcl(self, file_name="drawCylinders.tcl"):
        #raise NotImplementedError
        with open(file_name, 'w') as tclFile:
            tclFile.write("## beginning TCL script \n")
            def draw_cylinder(segment,radius_value=10,color="cyan"):
                tclFile.write("## cylinder being drawn... \n")
                r0 = segment.contour_to_position(0)
                r1 = segment.contour_to_position(1)
                
                radius_value = str(radius_value)
                color = str(color)
                
                tclFile.write("graphics top color {} \n".format(color))
                tclFile.write("graphics top cylinder {{ {} {} {} }} {{ {} {} {} }} radius {} resolution 30 filled yes \n".format(r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], radius_value))

            ## material
            tclFile.write("graphics top materials on \n")
            tclFile.write("graphics top material AOEdgy \n")
            
            ## iterate through the model segments
            for s in self.segments:
                if isinstance(s,DoubleStrandedSegment):
                    tclFile.write("## dsDNA! \n")
                    draw_cylinder(s,10,"cyan")
                elif isinstance(s,SingleStrandedSegment):
                    tclFile.write("## ssDNA! \n")
                    draw_cylinder(s,3,"orange")
                else:
                    raise Exception ("your model includes beads that are neither ssDNA nor dsDNA")
            ## tclFile complete
            tclFile.close()
