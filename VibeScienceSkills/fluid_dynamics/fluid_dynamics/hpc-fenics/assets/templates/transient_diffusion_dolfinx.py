from mpi4py import MPI
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl

domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
V = fem.functionspace(domain, ("Lagrange", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
u_n = fem.Function(V)
u_n.name = "u_n"

dt = 1.0e-2
T = 1.0
t = 0.0
f = fem.Constant(domain, 0.0)

facets = mesh.locate_entities_boundary(
    domain, domain.topology.dim - 1, lambda x: x[0] < 1.0e-12
)
dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, facets)
bc = fem.dirichletbc(0.0, dofs, V)

a = (u * v + dt * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
L = (u_n * v + dt * f * v) * ufl.dx

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_h = fem.Function(V)
u_h.name = "u"

with io.XDMFFile(domain.comm, "transient_diffusion.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    while t < T:
        u_h.x.array[:] = problem.solve().x.array
        xdmf.write_function(u_h, t)
        u_n.x.array[:] = u_h.x.array
        t += dt
