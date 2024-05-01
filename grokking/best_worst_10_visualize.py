from loss_contour_visualize import *
import argparse
import sys


if __name__ == '__main__':

    best_runs_id = ['1tmpr0ij', 'i11vuz8c', 'f7o6q7rv', 'i2td5wj8', 'u704bg05', 'fyj64sg3', 't8wb6zay', 'aksnn8hb', 'ikhv2x9n', 'efnutu2l']
    worst_runs_id = ['770wt2tg', '742kf75w', 'u64kpwd0', 'h25z2x6o', 'yddb7b23', '5o4jsmt1', 'm2tegcvn', '1vyham6z', '4wi88gom', '10u74oy8']

    runs_id = best_runs_id + worst_runs_id

    filenames = ['results_steps_100_range_1.json', 'results_steps_100_range_5.json',  'results_steps_100_range_10.json']

    for run_id in runs_id:
        for filename in filenames:
            run_path = 'huanran-research/grokking/' + run_id
            main(run_path, filename)
