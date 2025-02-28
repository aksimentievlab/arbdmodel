import gmsh
import numpy as np

def tetrahedron_volume(vertices):
    """Calculate volume of a tetrahedron given its vertices"""
    v0, v1, v2, v3 = vertices
    # Volume = |det(v1-v0, v2-v0, v3-v0)| / 6
    matrix = np.array([
        v1 - v0,
        v2 - v0,
        v3 - v0
    ])
    return abs(np.linalg.det(matrix)) / 6.0

def calculate_mesh_volume(mesh_file):
    """Calculate total volume of a tetrahedral mesh"""
    gmsh.initialize()
    
    try:
        # Load the mesh file
        gmsh.merge(mesh_file)
        
        # Get all tetrahedra (type 4 in Gmsh)
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)
        
        if not element_types:
            print("No volume elements found!")
            return 0.0
        
        total_volume = 0.0
        num_processed = 0
        
        # Process each type of 3D element
        for elem_type, elem_tags, nodes in zip(element_types, element_tags, node_tags):
            print(f"Processing element type {elem_type}")
            
            # Get nodes per element for this type
            num_nodes_per_element = len(nodes) // len(elem_tags)
            nodes = np.array(nodes).reshape(-1, num_nodes_per_element)
            
            # Process each element
            for element_nodes in nodes:
                # Get coordinates for each node
                vertices = []
                for node in element_nodes:
                    print(gmsh.model.mesh.getNode(node))
                    coord,_,_,_ = gmsh.model.mesh.getNode(node)
                    vertices.append(coord)
                
                # Calculate volume
                volume = tetrahedron_volume(np.array(vertices)*1e3)
                total_volume += volume
                
                num_processed += 1
                if num_processed % 100 == 0:
                    print(f"Processed {num_processed} elements. Current total volume: {total_volume}")
                
    finally:
        gmsh.finalize()
    
    return total_volume

def main():
    mesh_file = "3drod.msh"
    try:
        volume = calculate_mesh_volume(mesh_file)
        print(f"\nTotal mesh volume: {volume:.6f} cubic units")
        
    except Exception as e:
        print(f"Error calculating volume: {e}")
        raise e

if __name__ == "__main__":
    main()
