# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#pylint: disable=C0413
"""export stp files from CST"""
import os
import sys
import argparse

sys.path.append("/opt/cst/CST_Studio_Suite_2021/LinuxAMD64/python_cst_libraries/")  # path of python_cst_libraries

import cst.results
import cst
import cst.interface


def export_stp(cst_file, stp_path, json_path):
    """export stp files from CST"""
    project_name = cst_file

    current_de = cst.interface.DesignEnvironment()
    prj = current_de.open_project(project_name)
    current_de.set_quiet_mode = True

    vba_command = '\'#Language "WWB-COM"\n\n'
    vba_command += 'Option Explicit\n\n'
    vba_command += 'Sub Main\n'
    vba_command += '\tDim fn\n'
    vba_command += '\tfn = FreeFile()\n'
    vba_command += '\tOpen "' + json_path + '" For Output As #fn\n'
    vba_command += '\tPrint #fn, "{"\n'
    vba_command += '\tDim fqUnit, geoUnit, timeUnit\n'
    vba_command += '\tfqUnit = Units.GetFrequencyUnitToSI()\n'
    vba_command += '\tgeoUnit = Units.GetGeometryUnitToSI()\n'
    vba_command += '\ttimeUnit = Units.getTimeUnitToSI()\n'
    vba_command += '\tPrint #fn, """units"": {"\n'
    vba_command += '\tPrint #fn, "  ""geometry"": " + CStr(geoUnit) + ","\n'
    vba_command += '\tPrint #fn, "  ""time"": " + CStr(timeUnit) + ","\n'
    vba_command += '\tPrint #fn, "  ""frequency"": " + CStr(fqUnit)\n'
    vba_command += '\tPrint #fn, "},"\n'

    vba_command += '\tDim unit\n'
    vba_command += '\tunit = Units.GetGeometryUnitToSI()\n'
    vba_command += '\tPrint #fn, """ports"": ["\n'
    vba_command += '\tDim x0 As Double\n'
    vba_command += '\tDim x1 As Double\n'
    vba_command += '\tDim y0 As Double\n'
    vba_command += '\tDim y1 As Double\n'
    vba_command += '\tDim z0 As Double\n'
    vba_command += '\tDim z1 As Double\n'
    vba_command += '\tDim portNum\n'
    vba_command += '\tportNum = 0\n'
    vba_command += '\tDim portCount\n'
    vba_command += '\tportCount = 0\n'
    vba_command += '\tWhile portNum < 40\n'
    vba_command += '\t\tportNum = portNum + 1\n'
    vba_command += '\t\tOn Error GoTo tryNext\n'
    vba_command += '\t\tDiscretePort.GetLength(portNum)\n'
    vba_command += '\t\tDiscretePort.GetCoordinates(portNum, x0, y0, z0, x1, y1, z1)\n'
    vba_command += '\t\tIf portCount > 0 Then\n'
    vba_command += '\t\t\tPrint #fn, ","\n'
    vba_command += '\t\tEnd If\n'
    vba_command += '\t\tPrint #fn, "  {"\n'
    vba_command += '\t\tPrint #fn, "    ""src"": [" + CStr(x0 * unit) + ", " + CStr(y0 * unit) + ' \
                   '", " + CStr(z0 * unit) + "],"\n'
    vba_command += '\t\tPrint #fn, "    ""dst"": [" + CStr(x1 * unit) + ", " + CStr(y1 * unit) + ' \
                   '", " + CStr(z1 * unit) + "] "\n'
    vba_command += '\t\tPrint #fn, "  }"\n'
    vba_command += '\t\tportCount = portCount + 1\n'
    vba_command += '\t\ttryNext:\n'
    vba_command += '\tWend\n'
    vba_command += '\tPrint #fn, "],"\n'

    vba_command += '\tDim solidNum, solidName, i\n'
    vba_command += '\tDim idx_str As String\n'
    vba_command += '\tDim index As Integer\n'
    vba_command += '\tDim fileName As String\n'
    vba_command += '\tPrint #fn, """solids"": ["\n'
    vba_command += '\tfileName = "' + stp_path + '/solid"\n'
    vba_command += '\tsolidNum = Solid.GetNumberOfShapes\n'
    vba_command += '\tFor i = 0 To solidNum-1\n'
    vba_command += '\t\tsolidName = Solid.GetNameOfShapeFromIndex(i)\n'
    vba_command += '\t\tPrint #fn, "  {"\n'
    vba_command += '\t\tPrint #fn, "   ""index"": " + Cstr(i) + ","\n'
    vba_command += '\t\tPrint #fn, "   ""name"": """ + solidName + ""","\n'
    vba_command += '\t\tDim idx, comp, shape, file\n'
    vba_command += '\t\tidx = InStrRev(solidName, ":")\n'
    vba_command += '\t\tIf idx >= 0 Then\n'
    vba_command += '\t\t\tcomp = Left(solidName, idx - 1)\n'
    vba_command += '\t\t\tshape = Right(solidName, Len(solidName) - idx)\n'
    vba_command += '\t\t\tfile = fileName + "." + Cstr(i)\n'
    vba_command += '\t\t\tSelectTreeItem ("Components\\" + Replace(solidName, ":", "\\"))\n'
    vba_command += '\t\t\tWith STEP\n'
    vba_command += '\t\t\t\t.Reset\n'
    vba_command += '\t\t\t\t.FileName(file + ".stp")\n'
    vba_command += '\t\t\t\t.ExportAttributes(True)\n'
    vba_command += '\t\t\t\t.WriteSelectedSolids\n'
    vba_command += '\t\t\tEnd With\n'
    vba_command += '\t\t\tPrint #fn, "    ""stp"": """ + Replace(file + ".stp", "\\", "\\\\") + ""","\n'
    vba_command += '\t\tEnd If\n'
    vba_command += '\t\tPrint #fn, "    ""material"": """ + Solid.GetMaterialNameForShape(solidName) + """"\n'
    vba_command += '\t\tIf i < solidNum - 1 Then\n'
    vba_command += '\t\t\tPrint #fn, "  },"\n'
    vba_command += '\t\tElse\n'
    vba_command += '\t\t\tPrint #fn, "  }"\n'
    vba_command += '\t\tEnd If\n'
    vba_command += '\tNext\n'

    vba_command += '\tPrint #fn, "]"\n'
    vba_command += '\tPrint #fn, "}"\n'
    vba_command += '\tClose #fn\n'
    vba_command += 'End Sub\n'
    print('start')
    prj.schematic.execute_vba_code(vba_command)
    print('finish')
    prj.save()
    prj.close()
    current_de.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cst_path', type=str,
                        help='the path of cst file')
    parser.add_argument('--stp_path', type=str,
                        help='the path to save folder containing STP files')
    parser.add_argument('--json_path', type=str,
                        help='the path to save json file')

    opt = parser.parse_args()

    if not os.path.exists(opt.stp_path):
        os.makedirs(opt.stp_path)
    export_stp(opt.cst_path, opt.stp_path, opt.json_path)
