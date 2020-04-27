import numpy as np
import itertools
import sys

np.set_printoptions(threshold=sys.maxsize)


def add_to_list(list, a):
    for i, x in enumerate(list):
        if a[1] > x[1]:
            list.insert(i, a)
            return
    list.append(a)


def swap_edges_succ(path, e1, e2):
    a = np.where(path == e1[1])[0][0]
    b = np.where(path == e2[0])[0][0]
    if a > b:
        path = np.roll(path, -1 * (b + 2))
        a = np.where(path == e1[1])[0][0]
        b = np.where(path == e2[0])[0][0]
    path[a:b + 1] = np.flip(path[a:b + 1])
    return path


def calc_swap_edges_delta_succ(e1, e2, distances):
    old = distances[e1[0], e1[1]] + distances[e2[0], e2[1]]
    new = distances[e1[0], e2[0]] + distances[e2[1], e1[1]]
    return old - new


def calc_exchange_delta_succ(p, o, path, outside, distance):
    old = distance[path[p], path[p - 1]] + distance[path[p], path[(p + 1) % len(path)]]
    new = distance[outside[o], path[p - 1]] + distance[outside[o], path[(p + 1) % len(path)]]
    return old - new


def exchange_vertices_succ(path, outside, vp, vo):
    id_vp = np.where(path == vp)[0]
    id_vo = np.where(outside == vo)[0]
    path[id_vp], outside[id_vo] = outside[id_vo], path[id_vp]
    return path, outside


def LS_LM(distances):
    n_all = distances.shape[0]
    n = int(np.ceil(n_all / 2))

    path, outside = random_path(n, n_all)
    swap_actions = swap_edges_actions(n)
    exchange_actions = exchange_vertices_actions(path.shape[0], outside.shape[0])

    LM = []
    for i, s in enumerate(swap_actions):
        delta = calc_swap_edges_delta_succ((path[s[0] - 1], path[s[0]]), (path[s[1]], path[(s[1] + 1) % len(path)]), distances)
        if delta > 0:
            add_to_list(LM, ["s", delta, (path[s[0] - 1], path[s[0]]), (path[s[1]], path[(s[1] + 1) % len(path)])])

    for i, e in enumerate(exchange_actions):
        delta = calc_exchange_delta_succ(e[0], e[1], path, outside, distances)
        if delta > 0:
            add_to_list(LM, ["e", delta, path[e[0]], outside[e[1]]])

    id = 0
    while LM and id < len(LM):
        first = LM[id]

        if first[0] == "s":
            if first[2][0] in path and first[2][1] in path and first[3][0] in path and first[3][1] in path:
                if np.where(path == first[2][0])[0] == np.where(path == first[2][1])[0] - 1 and \
                        np.where(path == first[3][0])[0] == np.where(path == first[3][1])[0] - 1:
                    path = swap_edges_succ(path, first[2], first[3])
                    for i, x in enumerate(path):
                        x_next = path[(i + 1) % len(path)]
                        if x_next != first[2][0] and x != first[2][0] and x != first[3][1]:
                            delta = calc_swap_edges_delta_succ((first[2][0], first[3][1]), (x, x_next), distances)
                            if delta > 0:
                                add_to_list(LM, ["s", delta, (first[2][0], first[3][1]), (x, x_next)])

                        elif x_next != first[3][0] and x != first[3][0] and x != first[2][1]:
                            x_next = path[(i + 1) % len(path)]
                            delta = calc_swap_edges_delta_succ((first[3][0], first[2][1]), (x, x_next), distances)
                            if delta > 0:
                                add_to_list(LM, ["s", delta, (first[3][0], first[2][1]), (x, x_next)])
                    LM.pop(id)
                    id = 0
                else:
                    id += 1
            else:
                LM.pop(id)

        elif first[0] == "e":
            if first[2] in path and first[3] in outside:
                path, outside = exchange_vertices_succ(path, outside, first[2], first[3])

                id = 0
                for i, x in enumerate(path):
                    x_next = path[(i + 1) % len(path)]
                    if x_next != first[3] and x != first[3]:
                        out_next = path[(np.where(path == first[3])[0][0]) % len(path)]
                        if out_next != x:
                            delta = calc_swap_edges_delta_succ((first[3], out_next), (x, x_next), distances)
                            if delta > 0:
                                add_to_list(LM, ["s", delta, (first[3], out_next), (x, x_next)])
                for i, x in enumerate(outside):
                    index = (np.where(path == first[3])[0][0]) % len(path)
                    delta = calc_exchange_delta_succ(index, i, path, outside, distances)
                    if delta > 0:
                        add_to_list(LM, ["e", delta, first[3], x])
            LM.remove(first)

    return path


def random_path(length, dataset_size):
    points = np.arange(dataset_size)
    np.random.shuffle(points)
    path, outside = points[:length], points[length:]
    return path, outside


def swap_edges_actions(path_length):
    combinations = itertools.combinations(np.arange(path_length), 2)
    return np.array([[v1, v2] for v1, v2 in combinations if 1 < v2 - v1 < path_length - 1], 'int')


def exchange_vertices_actions(path_length, outside_length):
    product = itertools.product(np.arange(path_length), np.arange(outside_length))
    return np.array(list(product), 'int')


def calc_swap_edges_delta(swap_action, path, distances):
    v1, v2 = swap_action
    v1_prev = v1 - 1
    v2_next = (v2 + 1) % len(path)

    old = distances[path[v1_prev], path[v1]] + distances[path[v2], path[v2_next]]
    new = distances[path[v1_prev], path[v2]] + distances[path[v1], path[v2_next]]
    return old - new
