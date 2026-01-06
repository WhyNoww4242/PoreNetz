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

        self.points = np.array(mesh.points)
        self.tris = cell_dict[5]
        self.tets = cell_dict[10]


if __name__ == "__main__":
    cyl = FVMesh('shapes/cylinder.vtk')