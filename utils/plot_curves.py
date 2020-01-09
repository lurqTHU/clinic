import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os


def plot_curve(log_path, experiment_name, output):
    fp = open(log_path, 'r')

    curve_loss = []
    curve_acc = []

    for ln in fp:
        # get train_iterations and train_loss
        if 'Epoch[' in ln and 'Iteration[' in ln and 'Loss: ' in ln:
            loss = float(ln.split('Loss:')[1].split(',')[0])
            curve_loss.append(loss)

        if 'evaluation' in ln:
            acc = float(ln.split('Acc:')[1])
            curve_acc.append(acc)

    fp.close()

    plt.figure(figsize=(10,5))

    plt.subplot(1, 2, 1)
    plt.plot(curve_loss, 'b')
    plt.xlabel('iterations')
    plt.ylabel('train loss')

    plt.subplot(1, 2, 2)
    plt.plot(curve_acc, 'r')
    plt.xlabel('iterations')
    plt.ylabel('val accuracy')


    # plt.draw()
    fig_path = os.path.join(output, experiment_name + '.png')
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