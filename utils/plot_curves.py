import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
import os


def plot_curve(log_path, experiment_name, output_dir, trial_num=0):
    fp = open(log_path, 'r')

    train_loss = []
    val_loss = []
    curve_acc = []
    curve_dis = []
    curve_auc = []

    for ln in fp:
        # get train_iterations and train_loss
        if 'Epoch[' in ln and 'Iteration[' in ln and 'Loss: ' in ln:
            loss = float(ln.split('Loss:')[1].split(',')[0])
            train_loss.append(loss)

        if 'Mean Distance' in ln and 'Accuracy' in ln:
            acc = float(ln.split('Loss')[0].split(':')[-1].split(',')[0])
            curve_acc.append(acc)

            dis = float(ln.split('Mean Distance:')[1].split(',')[0])
            curve_dis.append(dis)
       
            loss = float(ln.split('Loss:')[1].split(',')[0])
            val_loss.append(loss)

        if 'AUC' in ln:
            auc = float(ln.split('AUC:')[1].split(',')[0])
            curve_auc.append(auc)

    fp.close()

    fig = plt.figure(figsize=(20,10))

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(train_loss, 'b', linewidth=3)
    ax.plot(val_loss, 'g', linewidth=3)
    ax.set_xlabel('Epochs', fontsize=30, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=30, fontweight='bold')
    ax.tick_params('both', labelsize=30)
#    ax.legend(['Train Loss', 'Validation Loss'], 
#               fontsize=30)
    ax.set_title('Learning Curve of QOL', fontsize=40, fontweight='bold')
#    ax.axvline(350, color='red', linewidth=3, linestyle='--')
#    ax.annotate('350', xy=(355, 0.57), fontsize=30, color='red')
   
#    plt.subplot(2, 2, 2)
#    plt.plot(curve_acc, 'r')
#    plt.xlabel('iterations')
#    plt.ylabel('val accuracy')
#
#    plt.subplot(2, 2, 3)
#    plt.plot(curve_dis, 'y')
#    plt.xlabel('epochs')
#    plt.ylabel('mean distance')
#
#    plt.subplot(2, 2, 4)
#    plt.plot(curve_auc, 'g')
#    plt.xlabel('epochs')
#    plt.ylabel('auc')

    # plt.draw()
    fig_path = os.path.join(output_dir, experiment_name \
               + 'trial_{}'.format(trial_num) + '.png')
    print('Min trainig loss: {:.3f}'.format(min(train_loss)))
    print('Max accuracy: {:.3f}'.format(max(curve_acc)))
    print('Min distance: {:.3f}'.format(min(curve_dis)))
    print('Max auc: {:.3f}'.format(max(curve_auc)))
    plt.savefig(fig_path)


def main():
    parser = argparse.ArgumentParser(description="plot training curves")
    parser.add_argument("--log_file", help="log file", type=str)
    parser.add_argument("--name", help="name of figure", type=str)
    parser.add_argument("--output_path", help="output path", type=str)
    args = parser.parse_args()

    plot_curve(args.log_file, args.name, args.output_path)


if __name__ == '__main__':
    main() 
