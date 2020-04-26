import numpy as np
import itertools
import functools
import sys
import utilites as ut

np.set_printoptions(threshold=sys.maxsize)


def add_to_list(list, a):
    for i, x in enumerate(list):
        if a[1] > x[1]:
            list.insert(i, a)
            return
    list.append(a)


def steepest_v2(distances, instance):
    n_all = distances.shape[0]
    n = int(np.ceil(n_all / 2))

    # Wygeneruj rozwiązanie x
    path, outside = random_path(n, n_all)
    path=np.array([10, 30, 40 , 60 , 80, 90])
    outside=np.array([15, 25, 35, 45, 55])
    n= len(path)
    swap_actions = swap_edges_actions(n)
    exchange_actions = exchange_vertices_actions(path.shape[0], outside.shape[0])

    # Zainicjuj LM – listę ruchów przynoszących poprawę uporządkowaną od najlepszego do najgorszego
    LM = []
    for i, s in enumerate(swap_actions):
        delta = calc_swap_edges_delta(s, path, distances)
        if delta > 0:
            add_to_list(LM, ["s", delta, s, path[s[0]], path[s[1]]])

    for i, e in enumerate(exchange_actions):
        delta = calc_exchange_delta(e, path, outside, distances)
        if delta > 0:
            add_to_list(LM, ["e", delta, e, path[e[0]], outside[e[1]]])
    # powtarzaj
    ut.print_plot(instance, 0, path, "LS")
    while LM:
        # print(len(LM))

        first = LM.pop(0)

        if first[0]== "s" and first[3] in path and first[4] in path:
            if first[1] == calc_swap_edges_delta(first[2], path, distances):
                print("--s------")
                swap_edges(path, first[2])
                print("swap:", first[2])

                for i, x in enumerate(path):
                    if first[2][0] - i == 1:
                        delta = calc_swap_edges_delta([first[2][0], i], path, distances)
                        if delta > 0 and first[2][1] != i and first[2][0] != i:
                            add_to_list(LM, ["s", delta, [first[2][0], i], path[first[2][0]], path[i], "s1"])

                    delta = calc_swap_edges_delta([first[2][1], i], path, distances)
                    if delta > 0 and first[2][1] != i and first[2][0] != i:
                        add_to_list(LM, ["s", delta, [first[2][1], i], path[first[2][1]], path[i], "s2"])

                print(first)
                print(len(LM), LM)
                print("path:",path)
                print("outside:",outside)
                ut.print_plot(instance, 0, path, "LS")

        elif first[0]== "e" and first[3] in path and first[4] in outside:
            if first[1] == calc_exchange_delta(first[2], path, outside, distances):
                print("--e------")
                exchange_vertices(path, outside, first[2])
                print("exchange:", first[2])
                for i, x in enumerate(outside):
                    delta = calc_exchange_delta([first[2][0], i], path, outside, distances)
                    if delta > 0 and first[2][0] != i:
                        add_to_list(LM, ["e", delta,[first[2][0], i], path[first[2][0]], outside[i], "e"])
                for i, x in enumerate(path):
                    delta = calc_swap_edges_delta([first[2][0], i], path, distances)
                    if delta > 0:
                        add_to_list(LM, ["s", delta, [first[2][0], i], path[first[2][0]], path[i], "se"])

                print(first)
                print(len(LM), LM)
                print("path:",path)
                print("outside:",outside)
                ut.print_plot(instance, 0, path, "LS")

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


def swap_edges(path, swap_action):
    print("before:", path)
    v1, v2 = swap_action

    path[v1:v2+1] = np.flip(path[v1:v2+1])

    print("after:",path)


def exchange_vertices(path, outside, exchange_action):
    print("before:",path)
    print("before:",outside)
    vp, vo = exchange_action
    path[vp], outside[vo] = outside[vo], path[vp]
    print("after:",path)
    print("after:",outside)

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

