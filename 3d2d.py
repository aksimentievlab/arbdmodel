import sys
import gmsh

gmsh.initialize(sys.argv)

gmsh.model.add("x7")
gmsh.open("3drod.msh")

print('Model ' + gmsh.model.getCurrent() + ' (' +
      str(gmsh.model.getDimension()) + 'D)')
entities = gmsh.model.getEntities()

for e in entities:
    # Dimension and tag of the entity:
    dim = e[0]
    tag = e[1]

    # Mesh data is made of `elements' (points, lines, triangles, ...), defined
    # by an ordered list of their `nodes'. Elements and nodes are identified by
    # `tags' as well (strictly positive identification numbers), and are stored
    # ("classified") in the model entity they discretize. Tags for elements and
    # nodes are globally unique (and not only per dimension, like entities).

    # A model entity of dimension 0 (a geometrical point) will contain a mesh
    # element of type point, as well as a mesh node. A model curve will contain
    # line elements as well as its interior nodes, while its boundary nodes will
    # be stored in the bounding model points. A model surface will contain
    # triangular and/or quadrangular elements and all the nodes not classified
    # on its boundary or on its embedded entities. A model volume will contain
    # tetrahedra, hexahedra, etc. and all the nodes not classified on its
    # boundary or on its embedded entities.

    # Get the mesh nodes for the entity (dim, tag):
    nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(dim, tag)

    # Get the mesh elements for the entity (dim, tag):
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, tag)

    # Elements can also be obtained by type, by using `getElementTypes()'
    # followed by `getElementsByType()'.

    # Let's print a summary of the information available on the entity and its
    # mesh.

    # * Type and name of the entity:
    type = gmsh.model.getType(dim, tag)
    name = gmsh.model.getEntityName(dim, tag)
    if len(name): name += ' '
    print("Entity " + name + str(e) + " of type " + type)

    # * Number of mesh nodes and elements:
    numElem = sum(len(i) for i in elemTags)
    print(" - Mesh has " + str(len(nodeTags)) + " nodes and " + str(numElem) +
          " elements")

    # * Upward and downward adjacencies:
    up, down = gmsh.model.getAdjacencies(dim, tag)
    if len(up):
        print(" - Upward adjacencies: " + str(up))
    if len(down):
        print(" - Downward adjacencies: " + str(down))

    # * Does the entity belong to physical groups?
    physicalTags = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
    if len(physicalTags):
        s = ''
        for p in physicalTags:
            n = gmsh.model.getPhysicalName(dim, p)
            if n: n += ' '
            s += n + '(' + str(dim) + ', ' + str(p) + ') '
        print(" - Physical groups: " + s)

    # * Is the entity a partition entity? If so, what is its parent entity?
    partitions = gmsh.model.getPartitions(dim, tag)
    if len(partitions):
        print(" - Partition tags: " + str(partitions) + " - parent entity " +
              str(gmsh.model.getParent(dim, tag)))

    # * List all types of elements making up the mesh of the entity:
    for t in elemTypes:
        name, dim, order, numv, parv, _ = gmsh.model.mesh.getElementProperties(
            t)
        print(" - Element type: " + name + ", order " + str(order) + " (" +
              str(numv) + " nodes in param coord: " + str(parv) + ")")



# Launch the GUI to see the model:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()
