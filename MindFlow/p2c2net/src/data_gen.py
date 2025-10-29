'''FD solver for 2d Buergers equation'''
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def apply_laplacian(mat, dx=1.0):
    # dx is inversely proportional to N
    """This function applies a discretized Laplacian
    in periodic boundary conditions to a matrix
    For more information see
    https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation_via_operator_discretization
    """

    # the cell appears 4 times in the formula to compute
    # the total difference
    neigh_mat = -5 * mat.copy()

    # Each direct neighbor on the lattice is counted in
    # the discrete difference formula
    neighbors = [
        (4 / 3, (-1, 0)),
        (4 / 3, (0, -1)),
        (4 / 3, (0, 1)),
        (4 / 3, (1, 0)),
        (-1 / 12, (-2, 0)),
        (-1 / 12, (0, -2)),
        (-1 / 12, (0, 2)),
        (-1 / 12, (2, 0)),
    ]

    # shift matrix according to demanded neighbors
    # and add to this cell with corresponding weight
    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0, 1))

    return neigh_mat / dx ** 2


def apply_dx(mat, dx=1.0):
    ''' central diff for dx'''

    # np.roll, axis=0 -> row
    # the total difference
    neigh_mat = -0 * mat.copy()

    # Each direct neighbor on the lattice is counted in
    # the discrete difference formula
    neighbors = [
        (1.0 / 12, (2, 0)),
        (-8.0 / 12, (1, 0)),
        (8.0 / 12, (-1, 0)),
        (-1.0 / 12, (-2, 0))
    ]

    # shift matrix according to demanded neighbors
    # and add to this cell with corresponding weight
    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0, 1))

    return neigh_mat / dx


def apply_dy(mat, dy=1.0):
    ''' central diff for dx'''

    # the total difference
    neigh_mat = -0 * mat.copy()

    # Each direct neighbor on the lattice is counted in
    # the discrete difference formula
    neighbors = [
        (1.0 / 12, (0, 2)),
        (-8.0 / 12, (0, 1)),
        (8.0 / 12, (0, -1)),
        (-1.0 / 12, (0, -2))
    ]

    # shift matrix according to demanded neighbors
    # and add to this cell with corresponding weight
    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0, 1))

    return neigh_mat / dy


def get_temporal_diff(u, v, r, dx):
    """
    Calculate temporal differences for the given arrays.
    
    Args:
        u: Input array for u component
        v: Input array for v component  
        r: Input array for r component
        
    Returns:
        Temporal differences for u and v components
    """

    laplace_u = apply_laplacian(u, dx)
    laplace_v = apply_laplacian(v, dx)

    u_x = apply_dx(u, dx)
    v_x = apply_dx(v, dx)

    u_y = apply_dy(u, dx)
    v_y = apply_dy(v, dx)

    # governing equation
    u_t = (1.0 / r) * laplace_u - u * u_x - v * u_y
    v_t = (1.0 / r) * laplace_v - u * v_x - v * v_y

    return u_t, v_t


def update(u0, v0, r=100.0, dt=0.05, dx=1.0):
    u_t, v_t = get_temporal_diff(u0, v0, r, dx)

    u = u0 + dt * u_t
    v = v0 + dt * v_t
    return u, v


def update_rk4(u0, v0, r=100.0, dt=0.05, dx=1.0):
    """Update with Runge-kutta-4 method
       See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """
    ############# Stage 1 ##############
    # compute the diffusion part of the update

    u_t, v_t = get_temporal_diff(u0, v0, r, dx)

    k1_u = u_t
    k1_v = v_t

    ############# Stage 1 ##############
    u1 = u0 + k1_u * dt / 2.0
    v1 = v0 + k1_v * dt / 2.0

    u_t, v_t = get_temporal_diff(u1, v1, r, dx)

    k2_u = u_t
    k2_v = v_t

    ############# Stage 2 ##############
    u2 = u0 + k2_u * dt / 2.0
    v2 = v0 + k2_v * dt / 2.0

    u_t, v_t = get_temporal_diff(u2, v2, r, dx)

    k3_u = u_t
    k3_v = v_t

    ############# Stage 3 ##############
    u3 = u0 + k3_u * dt
    v3 = v0 + k3_v * dt

    u_t, v_t = get_temporal_diff(u3, v3, r, dx)

    k4_u = u_t
    k4_v = v_t

    # Final solution
    u = u0 + dt * (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6.0
    v = v0 + dt * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0

    return u, v


def post_process(output, reso, xmin, xmax, ymin, ymax, num, fig_save_dir):
    ''' num: Number of time step
    '''

    x = np.linspace(0, reso, reso + 1)
    y = np.linspace(0, reso, reso + 1)
    x_star, y_star = np.meshgrid(x, y)
    x_star, y_star = x_star[:-1, :-1], y_star[:-1, :-1]

    u_pred = output[num, 0, :, :]
    v_pred = output[num, 1, :, :]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    cf = ax[0].scatter(x_star, y_star, c=u_pred, alpha=0.95, edgecolors='none', cmap='RdYlBu',
                       marker='s', s=3, vmin=-1, vmax=1)
    ax[0].axis('square')
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('black')
    # cf.cmap.set_over('whitesmoke')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('u-FDM')
    fig.colorbar(cf, ax=ax[0], fraction=0.046, pad=0.04)

    cf = ax[1].scatter(x_star, y_star, c=v_pred, alpha=0.95, edgecolors='none', cmap='RdYlBu',
                       marker='s', s=3, vmin=-1, vmax=1)
    ax[1].axis('square')
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('black')
    # cf.cmap.set_over('whitesmoke')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('v-FDM')
    fig.colorbar(cf, ax=ax[1], fraction=0.046, pad=0.04)

    # plt.draw()
    plt.savefig(f"{fig_save_dir}uv_{str(num).zfill(4)}.png")
    plt.close('all')


def rand_gaussian_ic(num_a, num_b, nx, ny, random_seed, plot=True):
    '''
    Generate random gaussian initial condition
    '''
    assert nx == ny
    x, y = [np.linspace(0, 1, nx + 1)] * 2
    xx, yy = np.meshgrid(x[:-1], y[:-1])
    # print("xx",xx)
    # print("yy",yy)
    wx, wy = [0] * 2
    np.random.seed(random_seed)#1
    # np.random.seed(2)#2
    # np.random.seed(3)#3
    # np.random.seed(4)#4
    ax = np.random.normal(0, 1, size=(num_a, num_b))
    bx = np.random.normal(0, 1, size=(num_a, num_b))
    ay = np.random.normal(0, 1, size=(num_a, num_b))
    by = np.random.normal(0, 1, size=(num_a, num_b))
    cxy = np.random.normal(-1, 1, size=2)
    for i in range(num_a):
        for j in range(num_b):
            wx = wx + ax[i, j] * np.sin(2 * np.pi * ((i - num_a // 2) * xx + (j - num_b // 2) * yy)) + bx[
                i, j] * np.cos(2 * np.pi * ((i - num_a // 2) * xx + (j - num_b // 2) * yy))
            wy = wy + ay[i, j] * np.sin(2 * np.pi * ((i - num_a // 2) * xx + (j - num_b // 2) * yy)) + by[
                i, j] * np.cos(2 * np.pi * ((i - num_a // 2) * xx + (j - num_b // 2) * yy))
    # print(cxy)
    # print(cxy[0])
    # print(cxy[1])
    ux = 2 * wx / wx.max() + cxy[0]
    uy = 2 * wy / wy.max() + cxy[1]

    if plot:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 8))
        fig.subplots_adjust(hspace=0.3, wspace=0.1)
        #
        ax[0].axis('square')
        cf = ax[0].contourf(xx, yy, ux, levels=101, cmap='jet')
        ax[0].set_xlim([0, 1])
        ax[0].set_ylim([0, 1])
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title(r'$u_0$', )
        fig.colorbar(cf, ax=ax[0], fraction=0.046, pad=0.04)
        #
        ax[1].axis('square')
        cf = ax[1].contourf(xx, yy, uy, levels=101, cmap='jet')
        ax[1].set_xlim([0, 1])
        ax[1].set_ylim([0, 1])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title(r'$v_0$')
        fig.colorbar(cf, ax=ax[1], fraction=0.046, pad=0.04)
        plt.show()

    return ux, uy
def createdata(ramdom_seed, n=104, n_simu_steps=2101, dt=0.001, re=200, data_save_dir='data/'):
    '''
    Create data for burgers equation
    '''
    dx = 1.0 / n

    print('seed = ', ramdom_seed)
    u, v = rand_gaussian_ic(
        num_a=10,
        num_b=10,
        nx=104,
        ny=104,
        random_seed=ramdom_seed,
        plot=False
    )

    u, v = u / 3.0, v / 3.0
    u_list = []
    v_list = []

    for step in range(n_simu_steps):
        u, v = update_rk4(u, v, re, dt, dx)  # [h,w]

        if (step + 1) % 1 == 0:
            print(step, '\n')
            u_list.append(u[None, ...])
            v_list.append(v[None, ...])

    u_record = np.concatenate(u_list, axis=0)  # [t,h,w]
    v_record = np.concatenate(v_list, axis=0)

    uv = np.concatenate((u_record[None, ...], v_record[None, ...]), axis=0)  # (c,t,h,w)
    uv = np.transpose(uv, [1, 0, 2, 3])  # (t,c,h,w) (751,2,128,128)
    scipy.io.savemat(data_save_dir + 'Burgers_2101x2x104x104_[RK4,R=200,dt=0_001,#seed'+str(ramdom_seed)+'].mat',
                     {'uv': uv})
    print("seed"+str(ramdom_seed)+"create over")

if __name__ == '__main__':
    # grid size
    for seed in range(1, 16):
        createdata(seed)
