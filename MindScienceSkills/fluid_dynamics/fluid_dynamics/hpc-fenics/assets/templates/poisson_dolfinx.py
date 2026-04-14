from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl

domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
V = fem.functionspace(domain, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, 1.0)

facets = mesh.locate_entities_boundary(
    domain, domain.topology.dim - 1, lambda x: x[0] < 1.0e-12
)
dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, facets)
bc = fem.dirichletbc(0.0, dofs, V)

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
)

uh = problem.solve()
