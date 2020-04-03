import numpy as np
import itertools
import sys

np.set_printoptions(threshold=sys.maxsize)


def steepest(distances, instance):
    n_all = distances.shape[0]
    n = int(np.ceil(n_all / 2))

    path, outside = random_path(n, n_all)

    # Generate actions
    swap_actions = swap_vertices_actions(n)
    exchange_actions = exchange_vertices_actions(path.shape[0], outside.shape[0])
    swap_delta = [cal_swap_delta(x[0], x[1], path, distances) for x in swap_actions]
    exchange_delta = [cal_exchange_delta(x[0], x[1], path, outside, distances) for x in exchange_actions]

    s_max = max(swap_delta)
    e_max = max(exchange_delta)
    if s_max > e_max:
        a = swap_delta.index(s_max)
        swap_vertices(path, swap_actions[a][0], swap_actions[a][1])
    else:
        a = exchange_delta.index(e_max)
        exchange_vertices(path, exchange_actions[a][0], outside, exchange_actions[a][1])

    while s_max > 0 and e_max > 0:
        swap_delta = [cal_swap_delta(x[0], x[1], path, distances) for x in swap_actions]
        exchange_delta = [cal_exchange_delta(x[0], x[1], path, outside, distances) for x in exchange_actions]

        s_max = max(swap_delta)
        e_max = max(exchange_delta)
        if s_max > e_max:
            a = swap_delta.index(s_max)
            swap_vertices(path, swap_actions[a][0], swap_actions[a][1])
        else:
            a = exchange_delta.index(e_max)
            exchange_vertices(path, exchange_actions[a][0], outside, exchange_actions[a][1])

    return path


def greedy(distances):
    n_all = distances.shape[0]
    n = int(np.ceil(n_all / 2))

    path, outside = random_path(n, n_all)

    # Generate actions
    swap_actions = swap_vertices_actions(n)
    exchange_actions = exchange_vertices_actions(path.shape[0], outside.shape[0])

    s_end = False
    e_end = False
    np.random.shuffle(swap_actions)
    np.random.shuffle(exchange_actions)
    s_idx=0
    e_idx=0

    while not s_end and not e_end:
            # wylosuj czy swap czy exchange
            if np.random.random() < 0.5 :
                if s_idx < len(swap_actions):
                    if cal_swap_delta(swap_actions[s_idx][0], swap_actions[s_idx][1], path, distances) > 0:
                        swap_vertices(path, swap_actions[s_idx][0], swap_actions[s_idx][1])
                        np.random.shuffle(swap_actions)
                        s_idx = 0
                    else:
                        s_idx += 1
                else:
                    e_end = True
            else:
                if e_idx < len(exchange_actions):
                    if cal_exchange_delta(exchange_actions[e_idx][0], exchange_actions[e_idx][1], path, outside, distances) > 0:
                        exchange_vertices(path, exchange_actions[e_idx][0], outside, exchange_actions[e_idx][1])
                        np.random.shuffle(exchange_actions)
                        e_idx = 0
                    else:
                        e_idx += 1
                else:
                    s_end=True

    return path


def random_path(length, dataset_size):
    points = np.arange(dataset_size)
    np.random.shuffle(points)
    path, outside = points[:length], points[length:]
    return path, outside


def swap_vertices(path, v1, v2):
    path[v1], path[v2] = path[v2], path[v1]


def exchange_vertices(path, vp, outside, vo):
    path[vp], outside[vo] = outside[vo], path[vp]


def swap_vertices_actions(path_length):
    return np.array(list(itertools.combinations(np.arange(path_length), 2)), "int64")


def exchange_vertices_actions(path_length, outside_length):
    product = itertools.product(np.arange(path_length), np.arange(outside_length))
    return np.array(list(product), "int64")


def cal_swap_delta(p1, p2, path, distance):
    old = distance[path[p1], path[p1 - 1]] + distance[path[p2], path[p2 - 1]] + \
          distance[path[p1], path[(p1 + 1) % len(path)]] + distance[path[p2], path[(p2 + 1) % len(path)]]
    new = distance[path[p2], path[p1 - 1]] + distance[path[p1], path[p2 - 1]] + \
          distance[path[p2], path[(p1 + 1) % len(path)]] + distance[path[p1], path[(p2 + 1) % len(path)]]

    if abs(p1 - p2) == 1 or abs(p1 - p2) == len(path) - 1:
        new += (distance[path[p1], path[p2]]) * 2

    return old - new


def cal_exchange_delta(p, o, path, outside, distance):
    old = distance[path[p], path[p - 1]] + distance[path[p], path[(p + 1) % len(path)]]
    new = distance[outside[o], path[p - 1]] + distance[outside[o], path[(p + 1) % len(path)]]
    return old - new
