
def write_cfg(fp, index):
    fp.write('[OEP]\n')
    fp.write('InputDensity                = none\n')
    fp.write('Structure                   = structure/H2/d%s.str\n' % (index))
    fp.write('OrbitalBasis                = aug-cc-pvqz\n')
    fp.write('PotentialBasis              = aug-cc-pvqz\n')
    fp.write('ReferencePotential          = hfx\n')
    fp.write('PotentialCoefficientInit    = zeros\n')
    fp.write('\n')
    fp.write('CheckPointPath              = oep-wy/chk/H2/d%s\n' % (index))
    fp.write('\n')
    fp.write('ConvergenceCriterion        = 1.e-12\n')
    fp.write('SVDCutoff                   = 5.e-6\n')
    fp.write('LambdaRegulation            = 0\n')
    fp.write('ZeroForceConstrain          = false\n')
    fp.write('RealSpaceAnalysis           = true\n')
    fp.write('\n\n')
    fp.write('[DATASET]\n')
    fp.write('MeshLevel                   = 3\n')
    fp.write('CubeLength                  = 0.9\n')
    fp.write('CubePoint                   = 9\n')
    fp.write('OutputPath                  = oep-wy/dataset/H2\n')
    fp.write('OutputName                  = d%s\n' % (index))
    fp.write('Symmetric                   = xz\n')
    fp.write('\n')


dis = range(500, 901, 40)
for d in dis:
    index = '%04d' % (d)
    with open('d%s.cfg' % (index), 'w') as fp:
        write_cfg(fp, index)
