

def create_dual_topology(p1,p2,u1=None,u2=None):
    """ p1 and p2 are parmed objects, u1,u2 are optional MDA universes bp restraints """
    assert(len(p1.residues) == len(p2.residues))

    i = p1.residues[5].atoms[0].idx
    assert( i == p2.residues[5].atoms[0].idx )

    assert( len(p2.urey_bradleys) == 0 )
    assert( len(p2.cmaps) == 0 )

    new = p1[:0]
    # new = read_files('tmp/unmutated.psf','tmp/unmutated.pdb')

    # p1_to_new = dict()
    p2_to_p1 = dict()
    u_p2_to_new = dict()
            
    bond_map = {a:[] for a in p2.atoms}
    angle_map = {a:[] for a in p2.atoms}
    dihed_map = {a:[] for a in p2.atoms}
    improper_map = {a:[] for a in p2.atoms}
    for b in p2.bonds:
        bond_map[b.atom1].append((b,0))
        bond_map[b.atom2].append((b,1))
    for b in p2.angles:
        angle_map[b.atom1].append((b,0))
        angle_map[b.atom2].append((b,1))
        angle_map[b.atom3].append((b,2))
    for b in p2.dihedrals:
        dihed_map[b.atom1].append((b,0))
        dihed_map[b.atom2].append((b,1))
        dihed_map[b.atom3].append((b,2))
        dihed_map[b.atom4].append((b,3))
    for b in p2.impropers:
        improper_map[b.atom1].append((b,0))
        improper_map[b.atom2].append((b,1))
        improper_map[b.atom3].append((b,2))
        improper_map[b.atom4].append((b,3))

    pairs = []

    last_p1_i = 0
    first_p1_atom = dict()     # keys are segids
    first_new_atom = dict()     # keys are segids
    last_p1_new_atom = dict()     # keys are segids
    processed_segments = set()
    for res1,res2 in zip(p1.residues,p2.residues):
        seg = res1.segid
        assert(seg == res2.segid)

        ## Add complete segment from p1 to new molecule
        if seg not in processed_segments:
            _next = [r.atoms[-1].idx+1 for r in p1.residues if r.segid == seg][-1]
            first_p1_atom[seg] = last_p1_i
            first_new_atom[seg] = len(new.atoms)
            new = new + p1[last_p1_i:_next]
            last_p1_new_atom[seg] = len(new.atoms)
            last_p1_i = _next
            processed_segments.add(seg)
            
        ## If residue is different in p1 and p2, add unique atoms from p2 to new 
        if (res1.name != res2.name):
            ## Perform a distance search to match atoms
            c1 = np.array([p1.coordinates[a.idx] for a in res1.atoms])
            c2 = np.array([p2.coordinates[a.idx] for a in res2.atoms])
            dists = distance_matrix(c1,c2)

            common_ids_j = set()
            common_atoms_j = set()
            for i,j in zip(*np.where(dists < 0.1)):            
                a1,a2 = res1.atoms[i],res2.atoms[j]
                p2_to_p1[a2] = a1
                if a1.name[0] == 'H': continue # add hydrogens later 
                if a1.name == a2.name and (a1.type == a2.type) and (a1.charge == a2.charge):
                    common_ids_j.add(j)
                    common_atoms_j.add(a2)

            for i,j in zip(*np.where(dists < 0.1)):            
                a1,a2 = res1.atoms[i],res2.atoms[j]
                if a1.name[0] != 'H': continue # only consider hydrogens
                if a1.name == a2.name and (a1.type == a2.type) and (a1.charge == a2.charge):
                    assert( len(a1.bonds) == 1 & len(a2.bonds) == 1 )
                    b1 = [a for a in (a1.bonds[0].atom1,a1.bonds[0].atom2) if a != a1][0]
                    b2 = [a for a in (a2.bonds[0].atom1,a2.bonds[0].atom2) if a != a2][0]
                    if b2 in common_atoms_j: # require that parent atoms of a1 and a2 are same to consider a match
                        common_ids_j.add(j)
                        common_atoms_j.add(a2)
                    
                    
            ## Problem with topology file or maybe psfgen causes H2' atoms to be out of place in mutated pdb, so H2' atoms fail distance search; here we work around the problem manually
            assert(any( a.name == "H2'" for a in res1.atoms ))
            assert(any( a.name == "H2'" for a in res2.atoms ))
            a1 = [a for a in res1.atoms if a.name == "H2'"][0]
            j,a2 = [x for x in enumerate(res2.atoms) if x[1].name == "H2'"][0]
            if a2 in p2_to_p1:
                assert(any( a.name == "H2'" for a in common_atoms_j ))
            else:
                p2_to_p1[a2] = a1
                common_ids_j.add(j)
                common_atoms_j.add(a2)
            assert(any( a.name == "H2'" for a in common_atoms_j ))

            ## Given list of common atoms, find remaining atoms unique to res2
            unique_atoms_j = list(set( res2.atoms[j]
                                for j in range(len(res2.atoms))
                                if j not in common_ids_j ))
            unique_ids_j = [a2.idx for a2 in unique_atoms_j]

            ## Finally, add unique atoms from res2 to new
            print(f'Adding partial residue: {len(new.bonds)} bonds + {len(p2[unique_ids_j].bonds)} new bonds')
            new = new + p2[unique_ids_j]
    print('Done adding atoms to new')

    ## Loop over residus in p1 and p2 a second time to add potentials (these are wiped by `new = new + p2[unique_ids_j]` above, so we can't do it there)
    current_seg = None
    current_new_atom = None
    for res1,res2 in zip(p1.residues,p2.residues):
        seg = res1.segid
        assert(seg == res2.segid)
        if current_seg != seg:
            current_seg = seg
            current_new_atom = last_p1_new_atom[seg]
            p1_to_new_di = first_new_atom[seg]-first_p1_atom[seg]

        if (res1.name != res2.name):
            c1 = np.array([p1.coordinates[a.idx] for a in res1.atoms])
            c2 = np.array([p2.coordinates[a.idx] for a in res2.atoms])
            dists = distance_matrix(c1,c2)

            common_ids_i = set()
            common_ids_j = set()
            common_atoms_j = set()
            for i,j in zip(*np.where(dists < 0.1)):
                a1,a2 = res1.atoms[i],res2.atoms[j]
                # p2_to_p1[a1] = a2
                p2_to_p1[a2] = a1
                if a1.name[0] == 'H': continue # consider hydrogens later
                if a1.name == a2.name and (a1.type == a2.type) and (a1.charge == a2.charge):
                    common_ids_i.add(i)
                    common_ids_j.add(j)
                    common_atoms_j.add(a2)

            for i,j in zip(*np.where(dists < 0.1)):            
                a1,a2 = res1.atoms[i],res2.atoms[j]
                if a1.name[0] != 'H': continue # only consider hydrogens
                if a1.name == a2.name and (a1.type == a2.type) and (a1.charge == a2.charge):
                    assert( len(a1.bonds) == 1 & len(a2.bonds) == 1 )
                    b1 = [a for a in (a1.bonds[0].atom1,a1.bonds[0].atom2) if a != a1][0]
                    b2 = [a for a in (a2.bonds[0].atom1,a2.bonds[0].atom2) if a != a2][0]
                    if b2 in common_atoms_j: # require that parent atoms of a1 and a2 are same to consider a match
                        common_ids_i.add(i)
                        common_ids_j.add(j)
                        common_atoms_j.add(a2)

                    
            ## H2' atoms are not where they should be in mutated pdb; working around problem with topology file or possibly psfgen by explicitly adding H2' to common atoms
            assert(any( a.name == "H2'" for a in res1.atoms ))
            assert(any( a.name == "H2'" for a in res2.atoms ))
            # assert(all( a.name != "H2'" for a in common_atoms_j ))
            i,a1 = [x for x in enumerate(res1.atoms) if x[1].name == "H2'"][0]
            j,a2 = [x for x in enumerate(res2.atoms) if x[1].name == "H2'"][0]
            if a2 in p2_to_p1: assert( p2_to_p1[a2] == a1 )
            p2_to_p1[a2] = a1
            common_ids_i.add(i)
            common_ids_j.add(j)
            common_atoms_j.add(a2)

            unique_atoms_j = list(set( res2.atoms[j]
                                for j in range(len(res2.atoms))
                                if j not in common_ids_j ))
            unique_ids_j = [a2.idx for a2 in unique_atoms_j]

            ## Find map of unique atoms from res2 to corresponding atoms added to new
            _num_new = len(unique_ids_j)
            _tmp = { a.name:a for a in new.atoms[current_new_atom:current_new_atom+_num_new] }
            current_new_atom += _num_new
            u_p2_to_new.update({ a2:_tmp[a2.name] for a2 in unique_atoms_j })
            
            ## Find nearest pairs of dual-topology atoms for EXB restraints, excluding H
            for i,a1 in enumerate(res1.atoms):
                if a1.name[0] == 'H':
                    dists[i,:] += 100
            for j,a2 in enumerate(res2.atoms):
                if a2.name[0] == 'H':
                    dists[:,j] += 100

            if len(c1) < len(c2):
                for i,a1 in enumerate(res1.atoms):
                    if i in common_ids_i: continue
                    if a1.name[0] == 'H': continue
                    j = np.argmin(dists[i])
                    pairs.append( (p1_to_new_di+a1.idx, u_p2_to_new[res2.atoms[j]].idx, dists[i,j]) )
            else:
                for j,a2 in enumerate(res2.atoms):
                    if j in common_ids_j: continue
                    if a2.name[0] == 'H': continue
                    i = np.argmin(dists[:,j])
                    pairs.append( (p1_to_new_di+res1.atoms[i].idx, u_p2_to_new[a2].idx, dists[i,j]) )
            
            ## set beta of FEP atoms
            for i,a in enumerate(new.atoms[p1_to_new_di+res1.atoms[0].idx:p1_to_new_di+res1.atoms[-1].idx+1]):
                if i in common_ids_i: continue
                a.bfactor = -1.0

            for _,a in u_p2_to_new.items():
                a.bfactor = 1.0
                                
            ## Add bonded potentials from p2 that include common atoms in new
            processed = set()       # avoid multiple entries of same bond
            for a2 in unique_atoms_j:
                for pot_map in (bond_map,angle_map,dihed_map,improper_map):
                    for b,rank in pot_map[a2]:
                        ## Avoid double-counting the potential
                        if b in processed: continue
                        processed.add(b)

                        ## Get p2 atoms in potential
                        if pot_map == bond_map:
                            atoms = (b.atom1,b.atom2)
                            gen_pot = lambda new_atoms: Bond(*new_atoms, type=b.type, order=b.order)
                            pot_list = new.bonds
                        elif pot_map == angle_map:
                            atoms = (b.atom1,b.atom2,b.atom3)
                            gen_pot = lambda new_atoms: Angle(*new_atoms, type=b.type)
                            pot_list = new.angles
                        elif pot_map == dihed_map:
                            atoms = (b.atom1,b.atom2,b.atom3,b.atom4)
                            gen_pot = lambda new_atoms: Dihedral(*new_atoms, type=b.type, improper=b.improper, ignore_end=b.ignore_end)
                            pot_list = new.dihedrals
                        elif pot_map == improper_map:
                            atoms = (b.atom1,b.atom2,b.atom3,b.atom4)
                            gen_pot = lambda new_atoms: Improper(*new_atoms, type=b.type)
                            pot_list = new.impropers
                        else:
                            raise Exception

                        ## skip if no atoms in bond are common atoms  
                        if all( a in unique_atoms_j for a in atoms ): continue

                        ## Find corresponding atoms in new 
                        new_atoms = [ u_p2_to_new[a] if a in unique_atoms_j
                                      else new.atoms[p1_to_new_di+p2_to_p1[a].idx]
                                      for a in atoms ]
                        assert( all( a in new.atoms for a in new_atoms ) )

                        ## Add potential to new.bonds, etc
                        pot_list.append( gen_pot(new_atoms) )
                        # print(f'... {len(new.bonds)} bonds of {len(new.bond_types)} types')

    bp_pairs = None
    def find_bp(u):
        # c,o = find_base_position_orientation( u )
        # bp = find_basepairs(u,c,o)
        ## Structure wasn't quite good enough for mrdna routines above to work, so we'll just walk through it knowing that it is balanced

        ## First find the nearest two strand termini
        seg_ends = [a.position for seg in u.segments
                    for a in [seg.atoms[0],seg.atoms[-1]]]
        seg_ends = np.array(seg_ends)

        dists = distance_matrix(seg_ends,seg_ends) + 100*np.eye(len(seg_ends))

        seg_end_map = {}
        for i,d in enumerate(dists):
            j = np.argmin(d)
            if i in seg_end_map:
                assert(seg_end_map[i] == j)
            seg_end_map[i] = j
            
        bp = -1 * np.ones(len(u.residues), dtype=int)
        processed = set()
        for I,J in seg_end_map.items():            
            if I in processed: continue
            processed.add(I)
            processed.add(J)

            reses1,reses2 = [u.segments[K//2].residues[::1-2*(K%2)] for K in (I,J)]
            for i,(r1,r2) in enumerate(zip(reses1,reses2)):
                if i == len(reses1)//2: break
                bp[r1.resindex] = r2.resindex
                bp[r2.resindex] = r1.resindex
        assert( np.all( bp >= 0 ) )
        return bp
    
    if u1 is not None and u2 is not None:
        bp_pairs = []
        bp1,bp2 = [find_bp(u) for u in (u1,u2)]
        for ra,(r1b,r2b) in enumerate(zip(bp1,bp2)):
            ## R1a pairs with R1b ; R2a pairs R2b
            R1a = p1.residues[ra]
            R2a = p2.residues[ra]
            R1b = p1.residues[r1b] if r1b >= 0 else None
            R2b = p2.residues[r2b] if r2b >= 0 else None
            a1i = b1i = None

            ## p1_to_new_di = first_new_atom[seg]-first_p1_atom[seg]
            ## pairs.append( (p1_to_new_di+res1.atoms[i].idx, u_p2_to_new[a2].idx, dists[i,j]) )
            
            if R1b is not None and ra < r1b:
                for n1,n2 in bp_bond_atoms[resname_to_key[R1a.name]]:
                    a = [x for x in R1a.atoms if x.name == n1][0]
                    b = [x for x in R1b.atoms if x.name == n2][0]

                    ## account for index offsets mapping p1 to new
                    seg = R1a.segid
                    ai = a.idx + first_new_atom[seg]-first_p1_atom[seg]
                    seg = R1b.segid
                    bi = b.idx + first_new_atom[seg]-first_p1_atom[seg]
                    bp_pairs.append((ai,bi))
                
            if R2b is not None and ra < r2b:
                ## skip redundant bps
                if R1b is not None and R1a.name == R2a.name and R1b.name == R2b.name: continue
                print(f'Not skipping {R2a.list.index(R2a)}--{R2b.list.index(R2b)}')
                for n1,n2 in bp_bond_atoms[resname_to_key[R2a.name]]:
                    a = [x for x in R2a.atoms if x.name == n1][0]
                    b = [x for x in R2b.atoms if x.name == n2][0] 
                    ## account for index offsets mapping p1 to new
                    try:
                        ai = u_p2_to_new[a].idx
                        assert( u_p2_to_new[a].name == n1 )
                    except:
                        seg = R1a.segid
                        a1 = [x for x in R1a.atoms if x.name == n1][0]
                        ai = a1.idx + first_new_atom[seg]-first_p1_atom[seg]
                    try:
                        bi = u_p2_to_new[b].idx
                        assert( u_p2_to_new[b].name == n2 )
                    except:
                        seg = R1b.segid
                        b1 = [x for x in R1b.atoms if x.name == n1][0]
                        bi = b1.idx + first_new_atom[seg]-first_p1_atom[seg]
                    bp_pairs.append((ai,bi))
    return new, pairs, bp_pairs

def convert_sod_to_mg(structure):
#     p solvent.residues[0].atoms[0].atom_type.__dict__.keys()
# dict_keys(['name', 'number', 'mass', 'atomic_number', 'epsilon', 'rmin', 'epsilon_14', 'rmin_14', 'nbfix', '_idx', '_bond_type', 'charge'])

    p = AmberParameterSet.from_leaprc('/data/server5/hchhabra/amber22/dat/leap/cmd/leaprc.water.opc')
    mg_t = p.atom_types['Mg2+']
    
    for a in structure:
        if a.residue.name == 'WAT':
            break
        elif a.residue.name != 'Na+':
            continue
        a.atom_type = mg_t
        a.residue.name = 'Mg2+'
        a.name = 'Mg2+'
        a.type = 'Mg2+'
        a.atomic_number = 12
        a._charge = 2
        a.charge = 2
        a.mass = 24.305

def make_mghh(structure):
    raise NotImplementedError
    p = AmberParameterSet.from_leaprc('/data/server5/hchhabra/amber22/dat/leap/cmd/leaprc.water.opc')
    mg_t = p.atom_types['Mg2+']
    
    for a in structure:
        if a.residue.name == 'WAT':
            break
        elif a.residue.name != 'Na+':
            continue
        a.atom_type = mg_t
        a.residue.name = 'Mg2+'
        a.name = 'Mg2+'
        a.type = 'Mg2+'
        a.atomic_number = 12
        a._charge = 2
        a.charge = 2
        a.mass = 24.305

        
        
if __name__ == '__main__':

    p1 = read_files('tmp/unmutated.psf','tmp/unmutated.pdb')
    p2 = read_files('tmp/mutated.psf','tmp/mutated.pdb')

    # new = read_files('tmp/unmutated.psf','tmp/unmutated.pdb')
    # new = new[:0]

    u1 = mda.Universe('tmp/unmutated.psf','tmp/unmutated.pdb')
    u2 = mda.Universe('tmp/mutated.psf','tmp/mutated.pdb')
    for a in u1.select_atoms("resname DT* and name C7").atoms:
        a.name = "C5M"
    for a in u2.select_atoms("resname DT* and name C7").atoms:
        a.name = "C5M"
    
    dual0,pairs,bp_pairs = create_dual_topology(p1,p2,u1,u2)
    dual0.save('tmp/dual.nosol.parmed.pdb')
    dual0.save('tmp/dual.nosol.parmed.psf', vmd=True)

    with open('output/dual.fep.exb','w') as fh:
        with open('output/dual.min.fep.exb','w') as fh2:
            for i,j,r0 in pairs:
                fh.write(f'bond {i} {j} 1 {r0}\n')
                fh2.write(f'bond {i} {j} 100 {r0}\n')

                
    with open('output/dual.bp.exb','w') as fh:
        for i,j in bp_pairs:
            fh.write(f'bond {i} {j} 1 2.8\n')

                
    # dual = load_file('output/dual.nosol.prmtop','output/dual.nosol.pdb')
    # solvent = load_file('tmp/solvent.prmtop','tmp/solvent.pdb')

    # convert_sod_to_mg(solvent)

    # combine = dual+solvent
    # # combine = dual
    # for a,b in zip(dual0.atoms,combine.atoms[:len(dual0.atoms)]):
    #     b.bfactor = a.bfactor

    # combine.save('output/dual.pdb')
    # combine.save('output/dual.prmtop')
    # combine.save('output/dual.rst7')
