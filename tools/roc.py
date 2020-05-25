import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
from sklearn import metrics


def calculate_best_point(fpr, tpr, thresholds):
    delta = np.abs(1 - tpr - fpr)
    ind = np.argmin(delta)
    return ind
    

def analyze_roc(fpr, tpr, thresholds, plot_path='./', 
                img_name='roc.png', target_name='VAS',
                plot=False):
    
    auc = metrics.auc(fpr, tpr)
    ind = calculate_best_point(fpr, tpr, thresholds)    

    if plot:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1,1,1)
        
        ax.plot(fpr, tpr, 'b', linewidth=2)
        ax.set_title('{:>3}'.format(target_name.capitalize()), 
                     fontsize=25, fontweight='bold')
        ax.set_xlabel('1-Specificity', fontsize=17, fontweight='bold')
        ax.set_ylabel('Sensitivity', fontsize=17, fontweight='bold')
      
        ax.set_aspect('equal')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params('both', labelsize=16)
        
        ax.plot(fpr[ind], tpr[ind], '.g', markersize=15)
        ax.annotate('Optimum Cutoff: {:.3f}\
                     \nSensitivity: {:.3f}\
                     \nSpecificity: {:.3f}'\
                     .format(thresholds[ind], tpr[ind],
                             1-fpr[ind]), xy=(fpr[ind], tpr[ind]),
                     arrowprops=dict(facecolor='black', width=0.5,
                                    headwidth=5,  shrink=0.1),
                     xytext=(fpr[ind]+0.1, tpr[ind]-0.2),
                    # horizontalalignment='left', verticalalignment='top',
                     fontsize=16, fontweight='bold')
    
        ax.annotate('AUC = {:.3f}'.format(auc), xy=(0.6, 0.2), fontsize=16, fontweight='bold')
        
        ax.add_line(Line2D((0.0, 1.0), (0.0, 1.0), linestyle='--', linewidth=2, color='gray'))    
    
        fig_path = os.path.join(plot_path, img_name)
        plt.savefig(fig_path)
    
    return auc, (tpr[ind], 1-fpr[ind], thresholds[ind])
