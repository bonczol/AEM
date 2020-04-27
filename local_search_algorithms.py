import numpy as np
import itertools
import functools
import sys

np.set_printoptions(threshold=sys.maxsize)


def steepest_vertices(distances):
    return steepest(distances, swap_vertices, calc_swap_vertices_delta, swap_vertices_actions)


def steepest_edges(distances):
    return steepest(distances, swap_edges, calc_swap_edges_delta, swap_edges_actions)


def greedy_vertices(distances):
    return greedy(distances, swap_vertices, calc_swap_vertices_delta, swap_vertices_actions)


def greedy_edges(distances):
    return greedy(distances, swap_edges, calc_swap_edges_delta, swap_edges_actions)


def steepest(distances, swap, calc_swap_delta, get_swap_actions):
    n_all = distances.shape[0]
    n = int(np.ceil(n_all / 2))

    path, outside = random_path(n, n_all)

    swap_actions = get_swap_actions(n)
    exchange_actions = exchange_vertices_actions(path.shape[0], outside.shape[0])

    while True: 
        swap_delta = [calc_swap_delta(s, path, distances) for s in swap_actions]
        exchange_delta = [calc_exchange_delta(e, path, outside, distances) for e in exchange_actions]

        smax_idx = np.argmax(swap_delta)
        emax_idx = np.argmax(exchange_delta)

        smax = swap_delta[smax_idx]
        emax = exchange_delta[emax_idx]

        if smax <= 0 and emax <= 0:
            break

        if smax > emax:
            swap(path, swap_actions[smax_idx])
        else:
            exchange_vertices(path, outside, exchange_actions[emax_idx])

    return path


def greedy(distances, swap, calc_swap_delta, get_swap_actions):
    n_all = distances.shape[0]
    n = int(np.ceil(n_all / 2))

    path, outside = random_path(n, n_all)

    swap_actions = get_swap_actions(n)
    exchange_actions = exchange_vertices_actions(path.shape[0], outside.shape[0])

    shuffle(swap_actions, exchange_actions)
    scurrent, ecurrent = 0, 0
    slen, elen = len(swap_actions), len(exchange_actions)

    while True:
        if scurrent >= slen and ecurrent >= elen:
            break
        elif (np.random.random() < 0.5 and scurrent < slen) or ecurrent >= elen:
            if calc_swap_delta(swap_actions[scurrent], path, distances) > 0:
                swap(path, swap_actions[scurrent])
                shuffle(swap_actions, exchange_actions)
                scurrent, ecurrent = 0, 0
            else:
                scurrent += 1
        else:
            if calc_exchange_delta(exchange_actions[ecurrent], path, outside, distances) > 0:
                exchange_vertices(path, outside, exchange_actions[ecurrent])
                shuffle(swap_actions, exchange_actions)
                scurrent, ecurrent = 0, 0
            else:
                ecurrent += 1

    return path


def random_path(length, dataset_size):
    points = np.arange(dataset_size)
    np.random.shuffle(points)
    path, outside = points[:length], points[length:]
    return path, outside


def swap_vertices(path, swap_action):
    v1, v2 = swap_action
    path[v1], path[v2] = path[v2], path[v1]


def swap_edges(path, swap_action):
    v1, v2 = swap_action
    path[v1:v2+1] = np.flip(path[v1:v2+1])


def exchange_vertices(path, outside, exchange_action):
    vp, vo = exchange_action
    path[vp], outside[vo] = outside[vo], path[vp]


def swap_vertices_actions(path_length):
    return np.array(list(itertools.combinations(np.arange(path_length), 2)), 'int')


def swap_edges_actions(path_length):
    combinations = itertools.combinations(np.arange(path_length), 2)
    return np.array([[v1,v2] for v1, v2 in combinations if 1 < v2 - v1 < path_length - 1 ], 'int')


def exchange_vertices_actions(path_length, outside_length):
    product = itertools.product(np.arange(path_length), np.arange(outside_length))
    return np.array(list(product), 'int')


def calc_swap_vertices_delta(swap_action, path, distance):
    p1, p2 = swap_action
    old = distance[path[p1], path[p1 - 1]] + distance[path[p2], path[p2 - 1]] + \
          distance[path[p1], path[(p1 + 1) % len(path)]] + distance[path[p2], path[(p2 + 1) % len(path)]]
    new = distance[path[p2], path[p1 - 1]] + distance[path[p1], path[p2 - 1]] + \
          distance[path[p2], path[(p1 + 1) % len(path)]] + distance[path[p1], path[(p2 + 1) % len(path)]]

    if abs(p1 - p2) == 1 or abs(p1 - p2) == len(path) - 1:
        new += (distance[path[p1], path[p2]]) * 2

    return old - new


def calc_swap_edges_delta(swap_action, path, distances):
    v1, v2 = swap_action
    v1_prev = v1 - 1
    v2_next = (v2 + 1) % len(path)

    old = distances[path[v1_prev], path[v1]] + distances[path[v2], path[v2_next]]
    new = distances[path[v1_prev], path[v2]] + distances[path[v1], path[v2_next]]
    return old - new


def calc_exchange_delta(exchange_action, path, outside, distance):
    p, o = exchange_action
    old = distance[path[p], path[p - 1]] + distance[path[p], path[(p + 1) % len(path)]]
    new = distance[outside[o], path[p - 1]] + distance[outside[o], path[(p + 1) % len(path)]]
    return old - new


def shuffle(swap_actions, exchange_actions):
    np.random.shuffle(swap_actions)
    np.random.shuffle(exchange_actions)
