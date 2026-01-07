import numpy as np
import pyvista as pv

class FVMesh:
    """
    Object used to build/update the mesh of pore network shape. Takes filename
    as input. Meant to be for pre-compute for the grid
    """
    def __init__(self, filename:str):

        print(f"Meshing {filename}...")

        # Read in Mesh Data
        self.filename = filename
        mesh = pv.read(filename)
        cell_dict = mesh.cells_dict 

        self.points = np.array(mesh.points)  # Locations of nodes
        self.tris = cell_dict[5]  # Stores triangles on surface
        self.tets = cell_dict[10]  # Stores volume tetrahedrons

        # Find boundary tetrahedrons
        self.bc_tets, self.bc_idx = self.BC_maker(self.tris, self.tets)

        self.tets_calc()

    def BC_maker(self, tris, tets):
        """
        Find tetrahedrons that share faces with boundary triangles.
        
        Parameters:
        -----------
        tris: nodes of surface triangles
        tets: nodes of volume tetrahedrons
        
        Returns:
        --------
        bc_tets, bc_idx: nodes of boundary tetrahedrons and their indices
        """
        bc_tets = []
        bc_idx = []
        
        # Loops of boundary triangles
        for i in range(0, len(tris)):
            tri_set = set(tris[i])
            
            # Loops of volume tetrahedrons
            for j in range(0, len(tets)):
                tet_set = set(tets[j])
                
                # Appends node set and index of BC tetrahedrons
                if tri_set.issubset(tet_set):
                    bc_tets.append(np.array(tet_set))
                    bc_idx.append(j)
        
        # Check for uniqueness
        unique_bc_tets = np.unique(bc_tets)
        
        if len(bc_tets) != len(unique_bc_tets):
            print(f"⚠️  Found duplicate boundary tetrahedrons!")
            print(f"   Total matches: {len(bc_tets)}")
            print(f"   Unique tets: {len(unique_bc_tets)}")
            print(f"   Duplicates: {len(bc_tets) - len(unique_bc_tets)}")
        else:
            print(f"✓ All ({len(unique_bc_tets)}) boundary tetrahedrons are unique")
        
        return unique_bc_tets, bc_idx
    
    def tets_calc(self):
        """
        Find tetrahedrons' centroid, volume 
        
        Parameters:
        -----------
        self: object 
        
        Returns:
        --------
        cell_volumes: volume of tetrahedrons
        cell_centroids: xyz location of tetrahedron centroids
        """
        # Extract 4 vertex coordinates
        p0 = self.points[self.tets[:,0]]
        p1 = self.points[self.tets[:,1]]
        p2 = self.points[self.tets[:,2]]
        p3 = self.points[self.tets[:,3]]

        # Edge vectors 
        v0 = p1 - p0
        v1 = p2 - p0
        v2 = p3 - p0
        v3 = p2 - p1
        v4 = p3 - p1
        v5 = p3 - p2

        # Scalar triple-product for volume...
        # V = abs(v1 \cdot (v2 \cross v3) / 6.0)                           
        cell_volumes = np.abs(np.sum(v0 * np.cross(v1, v2), axis=1))/6.0     

        # Get centroid (points to store concentration) 
        cell_centroids = (p0 + p1 + p2 + p3)/4.0

        # Centroid of the triangluar faces
        face0 = (p0 + p1 + p2)/3.0
        face1 = (p0 + p2 + p3)/3.0
        face2 = (p0 + p1 + p3)/3.0
        face3 = (p1 + p2 + p3)/3.0
        face_cat = np.concatenate((face0,face1,face2,face3))
        

        # normal of the triangluar
        face0norm = np.cross(v0, v1)
        face1norm = np.cross(v1, v3)
        face2norm = np.cross(v0, v3)
        face3norm = np.cross(v3, v4)

        # Computes face area using magnitude cross product
        face0_area = 0.5 * np.linalg.norm(face0norm, axis=1)
        face1_area = 0.5 * np.linalg.norm(face1norm, axis=1)
        face2_area = 0.5 * np.linalg.norm(face2norm, axis=1)
        face3_area = 0.5 * np.linalg.norm(face3norm, axis=1)
        

        print(len(face_cat),face_cat[:10])

        return cell_volumes, cell_centroids
    
if __name__ == "__main__":
    cyl = FVMesh('shapes/cylinder.vtk')