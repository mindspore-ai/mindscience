"""plot functions for phygeonet"""
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

from .py_mesh import visualize2d, set_axis_label


def plot_train(args, ev_hist, m_res_hist, time_spent):
    """plot train result"""
    plt.figure()
    plt.plot(m_res_hist, '-*', label='Equation Residual')
    plt.xlabel('Epoch')
    plt.ylabel('Residual')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'{args.figures_path}/convergence.pdf', bbox_inches='tight')
    tikzplotlib.save(f'{args.figures_path}/convergence.tikz')
    plt.figure()
    plt.plot(ev_hist, '-x', label=r'$e_v$')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'{args.figures_path}/error.pdf', bbox_inches='tight')
    tikzplotlib.save(f'{args.figures_path}/error.tikz')
    ev_hist = np.asarray(ev_hist)
    m_res_hist = np.asarray(m_res_hist)
    np.savetxt(f'{args.save_data_path}/ev_hist.txt', ev_hist)
    np.savetxt(f'{args.save_data_path}/m_res_hist.txt', m_res_hist)
    np.savetxt(f'{args.save_data_path}/time_spent.txt', np.zeros([2, 2]) + time_spent)


def plot_train_process(args, coord, epoch, ofv_sb, output_v):
    """plot train process"""
    fig1 = plt.figure()
    ax = plt.subplot(1, 2, 1)
    visualize2d(ax, (coord[0, 0, 1:-1, 1:-1].numpy(),
                     coord[0, 1, 1:-1, 1:-1].numpy(),
                     output_v[0, 0, 1:-1, 1:-1].numpy()), 'horizontal', [0, 1])
    set_axis_label(ax, 'p')
    ax.set_title('CNN ' + r'$T$')
    ax.set_aspect('equal')
    ax = plt.subplot(1, 2, 2)
    visualize2d(ax, (coord[0, 0, 1:-1, 1:-1].numpy(),
                     coord[0, 1, 1:-1, 1:-1].numpy(),
                     ofv_sb[1:-1, 1:-1]), 'horizontal', [0, 1])
    set_axis_label(ax, 'p')
    ax.set_aspect('equal')
    ax.set_title('FV ' + r'$T$')
    fig1.tight_layout(pad=1)
    fig1.savefig(f"{args.figures_path}/{epoch}T.pdf", bbox_inches='tight')
    plt.close(fig1)
