from loss_contour_calculate import *
import argparse
import sys


if __name__ == '__main__':

    best_runs_id = ['1tmpr0ij', 'i11vuz8c', 'f7o6q7rv', 'i2td5wj8', 'u704bg05', 'fyj64sg3', 't8wb6zay', 'aksnn8hb', 'ikhv2x9n', 'efnutu2l']
    worst_runs_id = ['770wt2tg', '742kf75w', 'u64kpwd0', 'h25z2x6o', 'yddb7b23', '5o4jsmt1', 'm2tegcvn', '1vyham6z', '4wi88gom', '10u74oy8']

    runs_id = best_runs_id + worst_runs_id

    for search_range in [1, 5, 10]:
        args = argparse.Namespace(
            run_path='huanran-research/grokking/' + runs_id[int(sys.argv[1])],
            steps=100,
            search_range=search_range,
            device='cuda',
            twin_model=0
        )
        print(args)
        main(args)
