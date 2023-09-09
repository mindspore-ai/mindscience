"""deeponet plot"""
import os.path

import matplotlib.pyplot as plt


def save_loss_fig(args, ii, losst, lossv, lossv0):
    """save loss figures"""
    fig = plt.figure()
    plt.semilogy(ii, losst, "r", label="Training loss")
    plt.semilogy(ii, lossv, "b", label="Test loss")
    plt.semilogy(ii, lossv0, "b", label="Test loss0")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title("Training and test")
    plt.legend()
    plt.savefig(f"{args.figures_path}/Training_test0.png", dpi=300)
    plt.tight_layout()
    plt.close(fig)
    fig = plt.figure()
    plt.semilogy(ii, losst, "r", label="Training loss")
    plt.semilogy(ii, lossv, "b", label="Test loss")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title("Training and test")
    plt.legend()
    plt.savefig(f"{args.figures_path}/Training_test.png", dpi=300)
    plt.tight_layout()
    plt.close(fig)


def plot_prediction(y_pred, y_test, args):
    """plot prediction"""
    fig = plt.figure()
    plt.plot(y_pred.asnumpy(), y_test, "r.", y_test, y_test, "b:")
    plt.savefig(os.path.join(args.figures_path, "prediction.png"), dpi=300)
    plt.close(fig)
