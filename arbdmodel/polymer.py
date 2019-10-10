class PolymerParticle(PointParticle):
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
