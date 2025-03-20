#!/usr/bin/python3

import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import OrderedDict
import itertools
import ast
import re


def check(name, layout):
    if len(layout) == 0:
        raise RuntimeError(f'layout `{name}`: empty')
    item_lengs = 2
    for item in layout:
        if len(item) != item_lengs:
            raise RuntimeError(f'layout `{name}`: basis vector must have 2 elements (i.e., [n,m] dims)')


def get_matrix(layout):
    num_cols = len(layout)
    num_rows = len(layout[0])
    matrix = []
    for r in range(num_rows):
        row = []
        for c in range(num_cols):
            row.append(layout[c][r])
        matrix.append(row)
    return matrix


def combine_matrices(matrices):
    num_rows = len(matrices[0])
    result = [[] for i in range(num_rows)]

    for matrix in matrices:
        for idx, row in enumerate(matrix):
            for element in row:
                result[idx].append(element)

    return result


def dot_product(vec, value):
    result = 0
    for idx in range(0, len(vec)):
        if value & (2**idx):
            result ^= vec[idx]
    return result


def vis(ax, results):
    n_axis = 0
    m_axis = 1

    n = results.T[n_axis]
    m = results.T[m_axis]

    ax.scatter(m, n, marker='s')


def process(name, layout, verbose):
    num_basis_vectors = len(layout)
    input_upper_bound = 2**num_basis_vectors

    if verbose:
        print(f'{name}: {num_basis_vectors=}; {input_upper_bound=}')

    matrix = get_matrix(layout)
    return matrix, num_basis_vectors, input_upper_bound


def main(args):
    matrices = []
    input_upper_bounds = []
    for name, layout in args.layouts.items():
        check(name, layout)
        matrix, _, input_upper_bound = process(name, layout, args.verbose)
        matrices.append(matrix)
        input_upper_bounds.append(input_upper_bound)

    matrix = combine_matrices(matrices)
    if args.verbose:
        print(matrix)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.invert_yaxis()
    ax.grid(which='both')

    shifts = [0]
    for idx in range(1, len(input_upper_bounds)):
        shifts.append(shifts[idx - 1] + int(np.log2(input_upper_bounds[idx - 1])))

    group_index = list(args.layouts.keys()).index(args.vis_group)
    upper_ranges = [list(range(ub)) for ub in input_upper_bounds[group_index + 1:]]
    for upper_coords in itertools.product(*upper_ranges):
        results = []
        lower_ranges = [list(range(ub)) for ub in input_upper_bounds[:group_index + 1]]
        for coords in itertools.product(*lower_ranges, *[[c] for c in upper_coords]):
            input_vector = 0
            for idx, coord in enumerate(coords):
                input_vector += (coord << shifts[idx])

            output = []
            for row in matrix:
                output.append(dot_product(row, input_vector))
            results.append(output)
        vis(ax, np.array(results))

    ax.set_aspect('equal', adjustable='box')
    plt.margins()
    plt.show()


def parse_layout(layout):
    register = re.search(re.compile(r'register\s+=\s+(.*?\]\])'), layout)
    if register is None:
        raise RuntimeError('failed to fine `register` dim while parsing `--layout` option')
    else:
        register = register.groups()[0]

    lane = re.search(re.compile(r'lane\s+=\s+(.*?\]\])'), layout)
    if lane is None:
        raise RuntimeError('failed to fine `lane` dim while parsing `--layout` option')
    else:
        lane = lane.groups()[0]

    warp = re.search(re.compile(r'warp\s+=\s+(.*?\]\])'), layout)
    if warp is None:
        raise RuntimeError('failed to fine `warp` dim while parsing `--layout` option')
    else:
        warp = warp.groups()[0]

    return ast.literal_eval(register), ast.literal_eval(lane), ast.literal_eval(warp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--register', default='[[0, 1], [0, 2], [0, 4]]', type=str,
                        help='vector basis for `register` dim')
    parser.add_argument('-l', '--lane', default='[[0, 8], [0, 16], [0, 32], [1, 0], [2, 0], [4, 0]]', type=str,
                        help='vector basis for `lane` dim')
    parser.add_argument('-w', '--warp', default='[[0, 64], [8, 0]]', type=str, help='vector basis for `warp` dim')
    parser.add_argument('--layout', type=str, help="linear layout string")
    parser.add_argument('--vis-group', choices=['register', 'lane', 'warp'], default='lane',
                        help='select group to highlight during visualization')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()

    if args.layout:
        register, lane, warp = parse_layout(args.layout)
    else:
        register = ast.literal_eval(args.register)
        lane = ast.literal_eval(args.lane)
        warp = ast.literal_eval(args.warp)

    args.layouts = OrderedDict()
    args.layouts['register'] = register
    args.layouts['lane'] = lane
    args.layouts['warp'] = warp

    if args.verbose:
        print(f'{register=}')
        print(f'{lane=}')
        print(f'{warp=}')

    main(args)
