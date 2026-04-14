from paraview.simple import *

reader = OpenDataFile("input.vtu")
view = GetActiveViewOrCreate("RenderView")

display = Show(reader, view)
ColorBy(display, ("POINTS", "pressure"))
view.ResetCamera()

SaveScreenshot("screenshot.png", view, ImageResolution=[1600, 900])
