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



if __name__ == "__main__":
    cyl = FVMesh('shapes/cylinder.vtk')