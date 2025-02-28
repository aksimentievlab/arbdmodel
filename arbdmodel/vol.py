import gmsh

gmsh.initialize()

gmsh.merge("3drod.msh")

gmsh.model.mesh.generate(3)

all_volumes = gmsh.model.getEntities(dim=3)
volume_group = gmsh.model.addPhysicalGroup(3, [vol[1] for vol in all_volumes])

gmsh.option.setNumber("Mesh.ComputeVolume", 1)
volume = gmsh.model.getPhysicalGroups()[0]
prop = gmsh.model.getPhysicalGroupProperties(volume[0], volume[1])
print(f"Volume: {prop[2]}")

gmsh.finalize()
