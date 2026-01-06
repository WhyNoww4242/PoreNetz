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

gmsh.write("shapes/cylinder.vtk")  # VTK format for PyVista
gmsh.write("shapes/cylinder.msh")  # VTK format for PyVista

# Get all surfaces
surfaces = gmsh.model.getEntities(2)    
surface_tags = [s[1] for s in surfaces]
print(surface_tags, surfaces)

gmsh.fltk.run()
gmsh.finalize()