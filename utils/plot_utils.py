import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_points(points, imp_points, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.', alpha=0.3, s=1)
    ax.scatter(imp_points[:, 0], imp_points[:, 1], imp_points[:, 2], c='r', marker='.', alpha=0.7, s=1)
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path, dpi=600)
    else:
        plt.savefig('img.jpg', dpi=600)

def plot_mol(points):
    """
    Schematic diagram of the molecular point cloud
    input:
        points (npoints, C)
    """
    _, C = points.shape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if C == 4:
        v = points[:, -1]
        min_v = min(v)
        max_v = max(v)
        color = [plt.get_cmap("seismic", 100)(int(float(i - min_v) / (max_v - min_v) * 100)) for i in v]
        plt.set_cmap(plt.get_cmap("seismic", 100))
        im = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker='.')
        fig.colorbar(im, format=matplotlib.ticker.FuncFormatter(lambda x, pos : round(x * (max_v - min_v) + min_v, 2)))
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.')
    plt.show()

def plot_loss(train_loss, valid_loss, save_path='./out/figs/trainLoss.png'):
    """Plot the training curve and the validation curve"""
    fig, axs = plt.subplots(figsize=(10, 5))
    axs.plot(train_loss, label='train loss')
    axs.plot(valid_loss, label='valid loss')
    axs.legend()
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Loss')
    plt.show()
    plt.savefig(save_path, dpi=600)

def plot_k_fold_roc_curve(FPR, TPR, AUC):
    """Plot the Receiver Operating Characteristic from a list
    of true positive rates and false positive rates."""
    
    # Initialize useful lists + the plot axes.
    tprs_interp = []
    #aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(12,11))
    
    # Plot ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(FPR, TPR)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = AUC[i]
        #aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1.2, alpha=0.7,
                 label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))
    
    # Plot the luck line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1.7, color='grey', alpha=.8)
    
    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(AUC)
    std_auc = np.std(AUC)
    ax.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2.2, alpha=1)
    
    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    # Fine tune and show the plot.
    ax.set_xlim([-0.04, 1.05])
    ax.set_ylim([-0.04, 1.05])
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    plt.savefig('./out/figs/cross-validation_roc_curve.png', dpi=600, bbox_inches='tight')
    return (f, ax)

def plot_true_pred_curve(y_true, y_pred):
    """Plot the regression prediction results"""
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, s=15, c='b', marker='s')
    ax.plot([-3, 2], [-3, 2], color='black', lw=1, linestyle='--')
    
    ax.set_xlabel('-LogAC50(Exp)')
    ax.set_ylabel('-LogAC50(Pred)')
    plt.savefig(f'./out/figs/logAC50Pred.jpg', dpi=600)
    return (fig, ax)