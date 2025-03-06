from arbdmodel.mesh_process_surface import*
from arbdmodel.mesh_process_volume import*
n=process_mesh_file("3drod.msh",density=19.3)

m= process_surface_mesh("Nanorod.msh", density=19.3)
