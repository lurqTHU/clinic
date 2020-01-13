import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os


def plot_curve(log_path, experiment_name, output):
    fp = open(log_path, 'r')

    curve_loss = []
    curve_acc = []
    curve_dis = []

    for ln in fp:
        # get train_iterations and train_loss
        if 'Epoch[' in ln and 'Iteration[' in ln and 'Loss: ' in ln:
            loss = float(ln.split('Loss:')[1].split(',')[0])
            curve_loss.append(loss)

        if 'Mean Distance' in ln and 'Accuracy' in ln:
            acc = float(ln.split('Loss')[0].split(':')[-1].split(',')[0])
            curve_acc.append(acc)

            dis = float(ln.split('Mean Distance:')[1].split(',')[0])
            curve_dis.append(dis)

    fp.close()

    plt.figure(figsize=(10,5))

    plt.subplot(2, 2, 1)
    plt.plot(curve_loss, 'b')
    plt.xlabel('iterations')
    plt.ylabel('train loss')

    plt.subplot(2, 2, 2)
    plt.plot(curve_acc, 'r')
    plt.xlabel('iterations')
    plt.ylabel('val accuracy')

    plt.subplot(2, 2, 3)
    plt.plot(curve_dis, 'y')
    plt.xlabel('epochs')
    plt.ylabel('mean distance')

    # plt.draw()
    fig_path = os.path.join(output, experiment_name + '.png')
    print('Min trainig loss: {:.3f}'.format(min(curve_loss)))
    print('Max accuracy: {:.3f}'.format(max(curve_acc)))
    print('Min distance: {:.3f}'.format(min(curve_dis)))
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
