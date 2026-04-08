from paraview.simple import *

reader = OpenDataFile("input.vtu")
slice_filter = Slice(Input=reader)
slice_filter.SliceType.Origin = [0.0, 0.0, 0.0]
slice_filter.SliceType.Normal = [1.0, 0.0, 0.0]

# Write as VTK PolyData (.vtp) to preserve geometry and topology.
# CSV export drops all cell connectivity and only emits point coordinates.
writer = CreateWriter("slice_output.vtp", slice_filter)
writer.UpdatePipeline()