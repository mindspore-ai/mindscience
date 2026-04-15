"""Define the boundary condition."""
import logging
import sympy
def dirichlet(n_dim, in_vars, out_vars):
    '''
        1d dirichlet boundary: a * u(x) = b
        2d dirichlet boundary: a * u(x, y) = b
        3d dirichlet boundary: a * u(x,y,z) = b
        note: a is a constant, the variables x, y, and z
              are in the boundary region, and b can be a function or a constant.
    '''
    bc_term = 1
    for var in in_vars:
        bc_term *= sympy.sin(4*sympy.pi*var)
    try:
        bc_term *= 1/(16*n_dim*sympy.pi*sympy.pi)
    except ZeroDivisionError:
        logging.error("Error: The divisor cannot be zero!")

    bc_eq = out_vars[0] - bc_term  # u(x) - bc_term
    equations = {"bc": bc_eq}
    return equations


def robin(n_dim, in_vars, out_vars):
    '''
        1d robin boundary: a * u(x) + b * u'(x) = c
        2d robin boundary: a * u(x, y) + b * u'(x, y) = c
        3d robin boundary: a * u(x,y,z) + b * u'(x,y,z) = c
        note: a, b is a constant, the variables x, y, and z
              are in the boundary region,
              u' is the number of external wizards of the function,
              and c can be a function or a constant.
    '''
    bc_term = 1
    u_x = 0
    bc_term_u = 0
    for var in in_vars:
        bc_term_ux = 1
        u_x += sympy.diff(out_vars[0], var)  # Derivation
        bc_term *= sympy.sin(4*sympy.pi*var)
        for i in in_vars:
            if i != var:  # Partial conduction
                bc_term_ux *= sympy.sin(4*sympy.pi*i)
            elif i == var:
                bc_term_ux *= sympy.cos(4*sympy.pi*i)
        bc_term_u += bc_term_ux
    try:
        bc_term *= 1/(16*n_dim*sympy.pi*sympy.pi)  # function
        bc_term_u *= 1/(4*n_dim*sympy.pi)
    except ZeroDivisionError:
        logging.error("Error: The divisor cannot be zero!")

    # u(x) + u'(x) - bc_term - bc_term_u
    bc_eq = out_vars[0] + u_x - bc_term - bc_term_u
    equations = {"bc": bc_eq}
    return equations


def periodic(n_dim, in_vars, out_vars):
    '''
        Periodic boundary conditions are a special case of Robin boundary conditions.
        1d periodic boundary: a * u(x) + b * u'(x) = a * u(x+T) + b * u'(x+T) = c
        2d periodic boundary: a * u(x,y) + b * u'(x,y) = a * u(x+T1,y+T2) + b * u'(x+T1,y+T2) = c
        3d periodic boundary: a * u(x,y,z) + b * u'(x,y,z) = a * u(x+T1,y+T2,z+T3) + b * u'(x+T1,y+T2,z+T3) = c
        note: a, b is a constant, the variables x, y, and z
              are in the boundary region,
              T1, T2, T3 corresponds to the period size of each variable in the defined interval,
              u' is the number of external wizards of the function,
              and c can be a function or a constant.
    '''
    _ = out_vars
    bc_term = 1
    for _ in in_vars:
        bc_term *= 2
    try:
        bc_term *= 1/(16*n_dim*sympy.pi*sympy.pi)
    except ZeroDivisionError:
        logging.error("Error: The divisor cannot be zero!")
    bc_eq = bc_term  # bc_term
    equations = {"bc": bc_eq}
    return equations


bc_type = {
    "dirichlet": dirichlet,
    "robin": robin,
    "periodic": periodic,
}


def get_bc(bc):
    '''return boundary condition'''
    try:
        boundary = bc_type[bc]
    except KeyError:
        logging.error("Wrong boundary name.")
    return boundary
