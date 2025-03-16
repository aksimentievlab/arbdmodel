from .core_objects import Transformable, Parent, Child, Clone
from .core_objects import ParticleType, PointParticle, RigidBodyType, RigidBody, Group
from .model import PdbModel, ArbdModel
from .engine import SimEngine, ArbdEngine, NamdEngine
from .sim_config import SimConf, DefaultSimConf
from .binary_manager import BinaryManager
from pathlib import Path
from parmed import load_file
from parmed.charmm import CharmmPsfFile
from .version import get_version
from .logger import logger,get_resource_path,devlogger
__version__ = get_version()

# Make everything available at package level
__all__ = ['Transformable', 'Parent', 'Child', 'Clone',
    'ParticleType', 'PointParticle', 'RigidBodyType', 'RigidBody', 'Group',
    'SimConf', 'DefaultSimConf', 'BinaryManager',
    'PdbModel', 'ArbdModel',
    'SimEngine', 'ArbdEngine', 'NamdEngine','get_resource_path']

def read_files(psf, pdb):
    """Read PSF and PDB files and combine them.
    
    Args:
        psf: Path to PSF file
        pdb: Path to PDB file
        
    Returns:
        Combined structure
    """
    p1 = CharmmPsfFile(psf)
    c1 = load_file(pdb)
    for a, b in zip(p1.atoms, c1.atoms):
        a.xx = b.xx 
        a.xy = b.xy
        a.xz = b.xz
        a.bfactor = b.bfactor
    return p1