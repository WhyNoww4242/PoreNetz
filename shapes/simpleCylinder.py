import gmsh
import pyvista as pv
import numpy as np

gmsh.initialize()
gmsh.model.add("porous_structure")

cyl1 = gmsh.model.occ.addCylinder(0,0,0, 5,5,10, 1 )

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)  # smallest allowed element
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)  # largest allowed element


gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)

gmsh.write("cylinder.msh")  # Gmsh format
gmsh.write("cylinder.vtk")  # VTK format for PyVista


gmsh.fltk.run()
gmsh.finalize()