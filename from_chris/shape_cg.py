import numpy as np

from parmed import load_file
from parmed.charmm import CharmmPsfFile
import MDAnalysis as mda
# from mdakit_sasa.analysis.sasaanalysis import SASAAnalysis
import freesasa 

from arbdmodel.shape_cg import find_shape_based_sites, get_particle_assignments
from mrrna.arbdmodel import Group, ParticleType, PointParticle
from mrrna.arbdmodel.interactions import HarmonicBond, NonbondedScheme

def read_files(psf,pdb):
    p1 = CharmmPsfFile(psf)
    c1 = load_file(pdb)
    for a,b in zip(p1.atoms,c1.atoms):
        a.xx = b.xx 
        a.xy = b.xy
        a.xz = b.xz
        a.bfactor = b.bfactor
    return p1


class ShapeCGNonbonded(NonbondedScheme):
    def __init__(self, debye_length=10, resolution=0.3, rMin=5):
        NonbondedScheme.__init__(self, typesA=None, typesB=None, resolution=resolution, rMin=rMin)
        self.debye_length = debye_length
        self.maxForce = 50

    def potential(self, r, typeA, typeB):
        """ Electrostatics """
        ld = self.debye_length 
        q1 = typeA.charge
        q2 = typeB.charge
        D = 80                  # dielectric of water
        ## units "e**2 / (4 * pi * epsilon0 AA)" kcal_mol
        A =  332.06371
        u_elec = (A*q1*q2/D)*np.exp(-r/ld) / r 
        
        epsilon = 0.5
        if 'nts' in typeA.__dict__ or 'nts' in typeB.__dict__:
            epsilon = 1.0

        # ## Derjaguin-type interaction for r6 vdW term
        # # https://doi.org/10.1063/5.0011446
        # H12 = ...
        # h = (r - typeA.sigma - typeB.sigma)
        # sigma_eff = typeA.sigma*typeB.sigma/(typeA.sigma+typeB.sigma)
        # f_derjaguin = - sigma_eff *H12 / (6*h**2) 
        # f_derjaguin[h<=0] = 0
        # u_derjaguin = np.empty( r.shape )
        # u_derjaguin[1:] = -np.cumsum(f_derjaguin*np.diff(r))
        # u_derjaguin[0] = u_derjaguin[1]
        # u_derjaguin = u_derjaguin - u_derjaguin[-1]
                   
        sigma = 0.5 * (typeA.sigma + typeB.sigma)

        r6 = (sigma/r)**6
        r12 = r6**2
        u_lj = 4 * epsilon * (r12-r6)

        ## modified to be repulsive
        rmin = sigma * 2**(1/6)
        i = np.where(r >= rmin)[0][0]
        u_lj = u_lj - u_lj[i]
        u_lj[i+1:] = 0
          
        u = u_elec + u_lj
        u[0] = u[1]             # Remove NaN

        maxForce = self.maxForce
        if maxForce is not None:
            assert(maxForce > 0)
            f = np.diff(u)/np.diff(r)
            f[f>maxForce] = maxForce
            f[f<-maxForce] = -maxForce
            u[0] = 0
            u[1:] = np.cumsum(f*np.diff(r))
        
        u = u-u[-1]
        return u

class ShapeCGFactory:

    def __init__(self, psf, pdb, disordered_residues = None,  name='SHCG'):
        self.psf = psf
        self.pdb = pdb
        self.name = name

        if disordered_residues is None:
            disordered_residues = []
        
        ## Using parmed because my version of MDA sometimes misreads psf attributes
        self._fine = _fine = read_files(psf,pdb)
        # self._fine_u = mda.Universe(psf,pdb)
        self._fine_positions = _fine.coordinates
        self._fine_masses = np.array([a.mass for a in _fine])
        self._fine_charges = np.array([a.charge for a in _fine])
        self._fine_names = np.array([a.name for a in _fine])
        self._fine_resnames = np.array([a.residue.name for a in _fine])
        self._fine_idp = np.array([1 if a.residue.idx in disordered_residues else 0 for a in _fine])
        self._fine_resid = np.array([a.residue.idx for a in _fine])
        self._fine_total_mass = self._fine_masses.sum()
        self._coarse_dict = {}  # dictionary that caches shape-CG model with number of beads as key

    def calc_atom_sasa0( self, atom_slice ):
        ## Calculate SASA in protein in usual way
        sl = atom_slice
        try:
            self.atom_radii
        except:
            structure = freesasa.Structure()
            structure.addAtoms( self._fine_names,
                                self._fine_resnames,
                                [int(x) for x in self._fine_resid],
                                len(self._fine_positions) * 'A',
                                *[self._fine_positions[:,i] for i in range(3)] )

            res = freesasa.calc(structure)

            self.atom_radii = np.array([structure.radius(i)+1.4 for i in range(res.nAtoms())])
            self.atom_sasa = np.array([res.atomArea(i) for i in range(res.nAtoms())])
            self.atom_area = 4*np.pi*self.atom_radii**2

        w = self.atom_sasa[sl]/self.atom_area[sl]
        # w = self.atom_sasa[sl]
        # return w > 0, self.atom_radii[sl]
        return w, self.atom_radii[sl]

    def calc_atom_sasa( self, atom_slice ):
        ## Calculate SASA in protein in usual way
        sl = atom_slice
        try:
            raise
            self.atom_radii
        except:
            structure = freesasa.Structure()
            structure.addAtoms( self._fine_names[sl],
                                self._fine_resnames[sl],
                                [int(x) for x in self._fine_resid[sl]],
                                len(self._fine_positions[sl]) * 'A',
                                *[self._fine_positions[sl][:,i] for i in range(3)] )

            res = freesasa.calc(structure)

            self._atom_radii = np.array([structure.radius(i) for i in range(res.nAtoms())])
            self._atom_sasa = np.array([res.atomArea(i) for i in range(res.nAtoms())])
            self._atom_area = 4*np.pi*self._atom_radii**2

        w = self._atom_sasa/self._atom_area
        # w = self.atom_sasa[sl]
        # return w > 0, self.atom_radii
        return w, self._atom_radii
        
    def get_coarse_protein( self, num_CG_sites = None ):
        if num_CG_sites is None:
            # Default to ~1000 dalton/site
            num_CG_sites = int(np.round(self._fine_total_mass/5000))
        
        if num_CG_sites not in self._coarse_dict:
            r_cg = find_shape_based_sites( self._fine_positions,
                                           N_cg = num_CG_sites,
                                           weights = self._fine_masses )

            # print(r_cg)
            mapping = get_particle_assignments( self._fine_positions, r_cg, max_distance=80 )
            self.mapping = mapping
            
            types = []
            parts = []

            cg_fine_resids = [self._fine_resid[mapping == i].mean() for i in range(num_CG_sites)]
            cg_resids = np.argsort(np.argsort(cg_fine_resids)) + 1

            def get_effective_radius(points, center, radii, weights=None):
                ## The following is taken from https://doi.org/10.1080/08927020701191349
                ## First column is cylinder aspect ratio, second column is 2nd virial = 0.5 <v_excl>
                ## For a sphere <v_excl> = 4 * 4|3 pi r**3

                if weights is None:
                    weights = np.ones(len(points))/len(points)
                
                ## SVD to get plane fitting to points
                U,s,V = np.linalg.svd(points-center[None,:])
                ax1,ax2,ax3 = V # ax3 is smallest axis
                projected = [np.einsum('ij,j->i', points-center[None,:], ax/np.linalg.norm(ax)) for ax in V]
                # m1,m2,m3 = [(weights*p**2).sum() for p in projected] # moments w.r.t. weights
                r1,r2,r3 = list(sorted([np.sqrt((weights*((p)**2+radii**2)).sum()/weights.sum()) for p in projected])) # r_gyr in inertia sense
                is_disc = (r1/r2) < (r2/r3)
                
                ## Approximate aspect ratio
                # aspect = np.sqrt(0.5*(r3**2+r2**2)) / np.sqrt(0.5*(r1**2+r2**2))
                aspect = r1/np.sqrt(0.5*(r2**2+r3**2)) if is_disc else r3/np.sqrt(0.5*(r2**2+r1**2))
                aspect = r1/r3 if is_disc else r3/r1

                # return np.sqrt((r3**2+r2**2+r1**2)/3)
                # return r3
                
                ## get cylinder data
                cylinder_data = """0.001 788.5
0.01 81.62
0.1 11.03
0.2 7.202
0.3 5.993
0.4 5.438
0.5 5.146
1.0 4.860
2.0 5.468
3.0 6.337
4.0 7.271
5.0 8.232
6.0 9.206
7.0 10.19
8.0 11.17
9.0 12.16
10 13.15
50 53.09
100 103.1"""
                cylinder_data = [[float(v) for v in line.split()] for line in cylinder_data.split('\n')]
                _A,_B = np.array(cylinder_data).T
                excl_vol = 2 * np.interp(aspect, _A, _B) * (2 * np.pi * r1 * r2 * r3)
                rad_eff = ((3/(np.pi*16))* excl_vol)**(1/3)
                # print('rgyr',[f'{v:.2f}' for v in (r1,r2,r3, r1/r2, r2/r3, aspect, rad_eff)])
                # import ipdb
                # ipdb.set_trace()
                ## For a sphere <v_excl> = 4 * 4|3 pi r**3
                return rad_eff
                
                
                # moment_ratio = 2*m3/(m1+m2) if is_disc else 0.5*(m3+m2)/m1

                ## For moment of inertia of a hollow (uncapped) cylinder, we have
                ##  `I_para / I_perp = (1/3) L**2/d**2`, where L is length and d is diameter
                ##  Hence
                
            
            part_by_resid = dict()
            for i,(r,rid) in sorted( enumerate(zip(r_cg, cg_resids)), key=lambda x: x[1][1] ):
                sl = (mapping == i)
                rad_gyr0 = np.sqrt( np.mean( ((self._fine_positions[sl] - r_cg[i])**2).sum(axis=-1) ) )
                # rad = np.sqrt(5/3) * ( rad_gyr )

                # sel = self._fine_u.atoms[sl]
                def _calc_sasa_rad0():
                    w,radii = self.calc_atom_sasa0( sl ) # weights
                    # w = w**(1/4)
                    # w = w > 0.1
                    # w = np.sqrt(w)
                    # print(w)
                    w = w/w.sum()                 # normalized
                    # return get_effective_radius( self._fine_positions[sl], r_cg[i], radii, w )
                    # assert( np.all(w < 1) )
                    # r_sq = (((self._fine_positions[sl] + radii[:,None]) - r_cg[i])**2).sum(axis=-1)
                    r_sq = ((self._fine_positions[sl] - r_cg[i])**2).sum(axis=-1) + radii**2
                    alpha = 1
                    return np.sqrt( (r_sq**alpha * w).sum() )**(1.0/alpha) # mean weighted by sasa fraction of atom
                def _calc_sasa_rad():
                    w,radii = self.calc_atom_sasa0( sl ) # weights
                    # w = w**(1/4)
                    # w = w > 0.9 * np.max(w)
                    # w = np.sqrt(w)
                    # print(w)
                    w = w/w.sum()                 # normalized
                    # return get_effective_radius( self._fine_positions[sl], r_cg[i], radii, w )
                    # assert( np.all(w < 1) )
                    # r_sq = (((self._fine_positions[sl] + radii[:,None]) - r_cg[i])**2).sum(axis=-1)
                    r_sq = ((self._fine_positions[sl] - r_cg[i])**2).sum(axis=-1) + radii**2
                    alpha = 1
                    return np.sqrt( (r_sq**alpha * w).sum() )**(1.0/alpha) # mean weighted by sasa fraction of atom
                _alpha = 1.1
                # _eps = 0.78*0.98*(18.06974932156109**(1.2-_alpha))
                _eps = 0.8* 12.78**(1.2-_alpha)
                rad_gyr = _eps*_calc_sasa_rad()**_alpha
                
                print(f'CG residue {i}/{num_CG_sites}: rgyr = {rad_gyr0}; rgyr_sasa = {rad_gyr} {rad_gyr/_calc_sasa_rad0()}')

                rad = rad_gyr
                mass = self._fine_masses[sl].sum() 
                charge = self._fine_charges[sl].sum()
                idp = self._fine_idp[sl].sum()

                a = (mass/self._fine_total_mass)**(1/3)
                D = num_CG_sites / a               # heuristic with correct scaling

                name = f'{self.name}_{num_CG_sites:02d}_{i:02d}'
                t = ParticleType(name,
                                 mass = mass,
                                 sigma = 2 * rad / 2**(1/6),
                                 charge = charge,
                                 idp = idp,
                                 diffusivity = D,
                                 # grid = [('../confine-300.dx', 1.0)],
                               )
                p = PointParticle(t, r, name, resid = rid, residue = rid)
                types.append(t)
                parts.append(p)
                part_by_resid[rid] = p
            g = Group(name=self.name, segname=self.name, children = parts)

            ## Add all-to-all elastic network for structured region
            num_idp = sum([p.idp > 0.5 for p in parts])        
            for i,p1 in enumerate(parts[:-1]):
                if p1.idp > 0.5: continue
                for j,p2 in enumerate(parts[i+1:],i+1):
                    if p2.idp > 0.5: continue
                    r0 = np.linalg.norm((p2.position - p1.position))
                    g.add_bond( i=p1, j=p2, bond = HarmonicBond( k = 50/(num_CG_sites-num_idp)**2, r0=r0 ), exclude = True )


            ## Add bonds to nearest beads for idp regions. Connect along
            ## nearest larger resid (no IDP on C-terminus)
            idp_parts = [p for p in parts if p.idp > 0.5]
            _idp_bonds = set()
            for p1 in idp_parts:
                try:
                    p2 = part_by_resid[p1.resid+1]
                    _idp_bonds.add( (p1,p2) )
                except:
                    pass
                try:
                    p2 = part_by_resid[p1.resid-1]
                    _idp_bonds.add( (p2,p1) )
                except:
                    pass
            for p1,p2 in _idp_bonds:
                r0 = np.linalg.norm((p2.position - p1.position))
                g.add_bond( i=p1, j=p2, bond = HarmonicBond( k = 1, r0=r0 ), exclude = True )
                
            self._coarse_dict[num_CG_sites] = (g,types)
        return self._coarse_dict[num_CG_sites][0]

    def get_coarse_types( self, num_CG_sites = None ):
        if num_CG_sites is None:
            # Default to ~1000 dalton/site
            num_CG_sites = int(np.round(self._fine_total_mass/5000))

        if num_CG_sites not in self._coarse_dict:
            self.get_coarse_protein(num_CG_sites)
        return self._coarse_dict[num_CG_sites][1]

    def generate_protein( self, position, orientation = None, index = 0, num_CG_sites = None ):
        # if orientation is not None:
        #     raise NotImplementedError

        new = self.get_coarse_protein( num_CG_sites ).duplicate()
        if orientation is not None:
            new.orientation = orientation
        new.position = position

        new.segname = f'{new.name}{index:03d}'
        new.name = new.segname
        # new.name = f'{chr(65+(index%24))}{index:03d}'
        return new

# if __name__ == '__main__':
#     pro = generate_protein( (0,0,0), num_CG_sites = 16 )
