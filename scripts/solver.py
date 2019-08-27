import argparse

import numpy as np


# -l 5 -w 4 -gl 0.01063 -gw 0.01069 -c 2 -k 5
parser = argparse.ArgumentParser('Compute delta given gradient')
parser.add_argument('-l', required=True, type=int, help='current #Depth')
parser.add_argument('-w', required=True, type=int, help='current #Width')
parser.add_argument('-gl', required=True, type=float, help='gradient of depth')
parser.add_argument('-gw', required=True, type=float, help='gradient of width')
parser.add_argument('-c', required=True, type=float, help='resource constrain')
parser.add_argument('-k', required=True, type=int, help='top-K results')
args = parser.parse_args()


if __name__ == "__main__":
    L, W = args.l, args.w
    grad_L, grad_W = args.gl, args.gw
    c = args.c
    k = args.k
    
    support_set = np.arange(0, 2 * float(c) / np.min((grad_L, grad_W)), 1.)
    solution_set = dict()
    for alpha in support_set:
        L_, W_ = L + alpha * grad_L, W + alpha * grad_W
        margin = np.abs(c - (L_ * (2. + W_)) / (L * (2. + W)))
        solution_set[margin] = alpha
    
    top_k = np.array(sorted(solution_set.keys()))[np.array(sorted(solution_set.keys())).argsort()[:5]]
    for rank, key in enumerate(top_k, 1):
        print("{}-closeness L1-loss {:.5f} alpha {:.2f} delta_L {:.2f} delta_W {:.2f}".format(
            rank, key, solution_set[key], solution_set[key] * grad_L, solution_set[key] * grad_W))
