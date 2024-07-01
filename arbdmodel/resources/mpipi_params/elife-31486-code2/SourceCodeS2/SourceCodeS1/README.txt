Script for generating PDB and contact annotations, as found in Source Data files S1 and S2.

(requires numpy and python 2.X)

USAGE:

   ./elife_write_PDBcontacts.py [pdb file]          -> All protein chains 
   ./elife_write_PDBcontacts.py [pdb file] [chain]  -> Specified chain 

ANNOTATION FORMAT:

   A.143.HIS*A.159.PHE         -> Contact between two sidechains (chain A, i. 143 and i. 159)

   A.98.TYR*A.99.VALTHR        -> Contact between sidechain at chain A and i. 98 to backbone
                                  peptide bond between i. 99 and i. 100

   A.208.ASPPRO*A.206.GLUALA   -> Planar contact between two backbone peptide bonds

   TOT.226 # CYS.1 MET.6 [...] -> Residue count numbers for observed amino acid types

