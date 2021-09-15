'#Language "WWB-COM"

Option Explicit

Sub Main
    Dim fn
    fn = FreeFile()
    Open "D:/tmp.json" For Output As #fn   'the path of json file
    Print #fn, "{"
    Dim fqUnit, geoUnit, timeUnit
    fqUnit = Units.GetFrequencyUnitToSI()
    geoUnit = Units.GetGeometryUnitToSI()
    timeUnit = Units.GetTimeUnitToSI()
    Print #fn, """freq"": [" + CStr(Solver.GetFmin() * fqUnit) + "," + CStr(Solver.GetFmax() * fqUnit) + "],"
    Print #fn, """units"": {"
    Print #fn, "  ""geometry"": " + CStr(geoUnit) + ","
    Print #fn, "  ""time"": " + CStr(timeUnit) + ","
    Print #fn, "  ""frequency"": " + CStr(fqUnit)
    Print #fn, "},"

    Dim unit
    unit = Units.GetGeometryUnitToSI()
    Print #fn, """ports"": ["
    Dim x0 As Double
    Dim x1 As Double
    Dim y0 As Double
    Dim y1 As Double
    Dim z0 As Double
    Dim z1 As Double
    Dim portNum
    portNum = 0
    Dim portCount
    portCount = 0
    While portNum < 40
        portNum = portNum + 1
        On Error GoTo tryNext
        DiscretePort.GetLength(portNum)
        DiscretePort.GetCoordinates(portNum, x0, y0, z0, x1, y1, z1)
        If portCount > 0 Then
            Print #fn, ","
        End If
        Print #fn, "  {"
        Print #fn, "    ""src"": [" + CStr(x0 * unit) + ", " + CStr(y0 * unit) + ", " + CStr(z0 * unit) + "],"
        Print #fn, "    ""dst"": [" + CStr(x1 * unit) + ", " + CStr(y1 * unit) + ", " + CStr(z1 * unit) + "] "
        Print #fn, "  }"
        portCount = portCount + 1
        tryNext:
    Wend
    Print #fn, "],"

    Dim solidNum, solidName, i
    Dim idx_str As String
    Dim index As Integer
    Dim fileName As String
    Print #fn, """solids"": ["
    fileName = "D:/tmp"   'the path of folder containing stp file
    solidNum = Solid.GetNumberOfShapes
    For i = 0 To solidNum - 1
        solidName = Solid.GetNameOfShapeFromIndex(i)
        SelectTreeItem ("Components\" + Replace(solidName, ":", "\"))
        Print #fn, "  {"
        Print #fn, "    ""index"": " + Cstr(i) + ","
        Print #fn, "    ""name"": """ + solidName + ""","
        Dim idx, comp, shape, file
        idx = InStrRev(solidName, ":")
        If idx >= 0 Then
            comp = Left(solidName, idx - 1)
            shape = Right(solidName, Len(solidName) - idx)
            file = fileName + "/" + Cstr(i)
            With STEP
                .Reset
                .FileName(file + ".stp")
                .ExportAttributes(True)
                .WriteSelectedSolids     'change to .Write(solidName) in version 2019
            End With
            Print #fn, "    ""stp"": """ + Replace(file + ".stp", "\\", "\\\\") + ""","
        End If
        Print #fn, "    ""material"": """ + Solid.GetMaterialNameForShape(solidName) + """"
        If i < solidNum - 1 Then
            Print #fn, "  },"
        Else
            Print #fn, "  }"
        End If
    Next

    Print #fn, "]"
    Print #fn, "}"
    Close #fn

End Sub
