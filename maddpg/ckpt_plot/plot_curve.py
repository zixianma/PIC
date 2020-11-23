import os
import sys
import numpy as np

import matplotlib as mpl
import csv
import argparse

mpl.use('Agg')
from matplotlib import pyplot as plt


def avg_list(l, avg_group_size=2):
    ret_l = []
    n = len(l)
    h_size = avg_group_size / 2
    for i in range(n):
        left = int(max(0, i - h_size))
        right = int(min(n, i + h_size))

        ret_l.append(np.mean(l[left:right]))
    return ret_l


def plot_result(t1, r1, fig_name, x_label, y_label, save_path):
    plt.close()
    base = None
    base, = plt.plot(t1, avg_list(r1))


    plt.grid()
    #plt.legend([base, teach1, teach2, teach3], ['CA error < 5%', 'CA error < 10%', 'CA error < 20%', 'MADDPG + global count'])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_name)

    try:
        plt.savefig(save_path + '.pdf')
        plt.savefig(save_path + '.png')
    except:
        print('ERROR:', sys.exc_info()[0])
        print('Terminate Program')
        sys.exit()
    print('INFO: Wrote plot to ' + save_path + '.pdf')


def plot_result2(t1, r1, r2, fig_name, x_label, y_label):
    plt.close()
    base = None
    l1, = plt.plot(t1, r1)
    l2, = plt.plot(t1, r2)

    plt.grid()
    plt.legend([l1, l2], ['train', 'val'])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_name)

    try:
        plt.savefig(fig_name + '.pdf')
        plt.savefig(fig_name + '.png')
    except:
        print('ERROR:', sys.exc_info()[0])
        print('Terminate Program')
        sys.exit()
    print('INFO: Wrote plot to ' + fig_name + '.pdf')


def plot_result_mul(fig_name, x_label, y_label, legend, save_path, t1, r1,  t2, r2, t3, r3):
    plt.close()

    l1, = plt.plot(t1, r1)
    if t2 is not None:
        l2, = plt.plot(t2, r2)
    if t3 is not None:
        l3, = plt.plot(t3, r3)
    plt.grid()
    plt.legend([l1, l2, l3], legend)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_name)

    try:
        plt.savefig(save_path + '.pdf')
        plt.savefig(save_path + '.png')
    except:
        print('ERROR:', sys.exc_info()[0])
        print('Terminate Program')
        sys.exit()
    print('INFO: Wrote plot to ' + save_path + '.pdf')


def read_csv(csv_path):
    res = {}
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            res[row[0]] = [float(r) for r in row[1:]]
    return res

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True,help='path to the train_curve.csv file')
    # parser.add_argument('--csv_path_2', required=True,help='path to the train_curve.csv file')
    # parser.add_argument('--csv_path_3', required=True,help='path to the train_curve.csv file')
    args = parser.parse_args()
    res = read_csv(args.csv_path)
    # res2 = read_csv(args.csv_path_2)
    # res3 = read_csv(args.csv_path_3)
    start = args.csv_path.find('exp_data') 
    start = start + 9 if start >= 0 else 0
    end = args.csv_path.rfind('/') 
    exp_name = args.csv_path[start:end]
    save_path = os.path.join(args.csv_path[:end], exp_name)
    if 'simple_tag' in exp_name:
        plot_result(res['steps'], res['rewards'], exp_name, 'steps', 'rewards', save_path + '_rewards')
        plot_result(res['steps'], res['collisions'], exp_name, 'steps', 'predator-prey collisions', save_path + '_collisions')
        plot_result(res['steps'], res['dists'], exp_name, 'steps', 'predator-prey min_dist', save_path + '_min_dist')
    elif 'simple_coop_push' in exp_name:
        plot_result(res['steps'], res['rewards'], exp_name, 'steps', 'rewards', save_path + '_rewards')
        plot_result(res['steps'], res['collisions'], exp_name, 'steps', 'agent-ball collisions', save_path + '_collisions')
        plot_result(res['steps'], res['avg_dists'], exp_name, 'steps', 'ball-target avg_dist', save_path + '_avg_dist')
        plot_result(res['steps'], res['occupied_targets'], exp_name, 'steps', 'num of occupied target', save_path + '_occupied_target')
    # Plot multiple results in one plot
    # def find_exp_name(path):
    #     start = path.find('exp_data') 
    #     start = start + 9 if start >= 0 else 0
    #     end = path.rfind('/') 
    #     return path[start:end]

    # exp_name = find_exp_name(args.csv_path)
    # exp_name2 = find_exp_name(args.csv_path_2)
    # exp_name3 = find_exp_name(args.csv_path_3)
    
    # legend = [exp_name, exp_name2, exp_name3]
    # if 'simple_tag' in exp_name:
    #     save_path = os.path.join('../../exp_data/multi', 'simple_tag')
    #     plot_result_mul('simple_tag', 'steps', 'rewards', legend, save_path + '_rewards', res['steps'], res['rewards'], res2['steps'], res2['rewards'], res3['steps'], res3['rewards'])
    #     plot_result_mul('simple_tag', 'steps', 'predator-prey collisions', legend, save_path + '_collisions', res['steps'], res['collisions'], res2['steps'], res2['collisions'], res3['steps'], res3['collisions'])
    #     plot_result_mul('simple_tag', 'steps', 'predator-prey min_dist', legend, save_path + '_min_dist', res['steps'], res['dists'], res2['steps'], res2['dists'], res3['steps'], res3['occupied_dists'])
    # elif 'simple_coop_push' in exp_name:
    #     save_path = os.path.join('../../exp_data/multi', 'simple_coop_push')
    #     plot_result_mul('simple_coop_push','steps', 'rewards', legend, save_path + '_rewards', res['steps'], res['rewards'], res2['steps'], res2['rewards'], res3['steps'], res3['rewards'])
    #     plot_result_mul('simple_coop_push','steps', 'agent-ball collisions', legend, save_path + '_collisions', res['steps'], res['collisions'], res2['steps'], res2['collisions'], res3['steps'], res3['collisions'])
    #     plot_result_mul('simple_coop_push','steps', 'ball-target avg_dist', legend, save_path + '_avg_dist', res['steps'], res['avg_dists'], res2['steps'], res2['avg_dists'], res3['steps'], res3['avg_dists'])
    #     plot_result_mul('simple_coop_push','steps', 'num of occupied target', legend, save_path + '_occupied_target', res['steps'], res['occupied_targets'], res2['steps'], res2['rewards'], res3['steps'], res3['rewards'])
        
if __name__ == "__main__":
    main()

