import numpy as np
import utilites as ut
import local_search_algorithms as ls
import time
import greedy_algorithms as greedy

AVG_MSLS_SCORE = 15433.7
AVG_MSLS_TIME = 200

def multiple_start_ls(distances):
    m = 100
    best_path = ls.steepest_edges(distances)
    best_score = ut.evaluate(best_path, distances)

    print("best: ", best_score)

    for i in range(m - 1):
        result_path = ls.steepest_edges(distances)
        result_score = ut.evaluate(result_path, distances)

        print(i, "best: ", best_score, "result: ", result_score)
        if result_score < best_score:
            best_path = result_path
            best_score = result_score

    return best_path


def iterated_ls1(distances):
    n_all = distances.shape[0]
    n = int(np.ceil(n_all / 2))

    path, outside = ls.random_path(n, n_all)

    swap_actions = ls.swap_edges_actions(n)
    exchange_actions = ls.exchange_vertices_actions(path.shape[0], outside.shape[0])
    start = time.perf_counter()

    while True:
        path = steepest(distances, path, outside, swap_actions, exchange_actions)
        score = ut.evaluate(path, distances)
    # Tutaj nie chodzi o wynik tylko to, że czas musi być mniejszy niż średni dla MLS
    # if score < AVG_MSLS_SCORE:
        if time.perf_counter() - start > AVG_MSLS_TIME:
            break
        else:
            path = perturbation_small(path)

    return path


def iterated_ls2(distances):
    n_all = distances.shape[0]
    n = int(np.ceil(n_all / 2))

    path, outside = ls.random_path(n, n_all)

    swap_actions = ls.swap_edges_actions(n)
    exchange_actions = ls.exchange_vertices_actions(path.shape[0], outside.shape[0])
    start = time.perf_counter()
    path = steepest(distances, path, outside, swap_actions, exchange_actions)
    path = path.tolist()
    while time.perf_counter() - start < AVG_MSLS_TIME:
        path = perturbation_destroy(path)
        path = greedy_cycle(path, n, n_all, distances)

    return path


def perturbation_small(path):
    max_offset = int(path.shape[0] / 4)

    p1 = np.random.randint(max_offset)
    p2 = p1 + 1 + np.random.randint(max_offset)
    p3 = p2 + 1 + np.random.randint(max_offset)
    return np.concatenate((path[:p1], path[p3:], path[p2:p3], path[p1:p2]))


def perturbation_destroy(path):
    # TUTAJ Trzeba jakos sensownie wywalic z 20% wiercholkow z rozwiazania
    a = int(len(path) * 0.2)
    for i in range(1,a):
        path.pop(np.random.randint(len(path)-i))
    return path


def steepest(distances, path, outside, swap_actions, exchange_actions):
    while True:
        swap_delta = [ls.calc_swap_edges_delta(s, path, distances) for s in swap_actions]
        exchange_delta = [ls.calc_exchange_delta(e, path, outside, distances) for e in exchange_actions]

        smax_idx = np.argmax(swap_delta)
        emax_idx = np.argmax(exchange_delta)

        smax = swap_delta[smax_idx]
        emax = exchange_delta[emax_idx]

        if smax <= 0 and emax <= 0:
            break

        if smax > emax:
            ls.swap_edges(path, swap_actions[smax_idx])
        else:
            ls.exchange_vertices(path, outside, exchange_actions[emax_idx])

    return path


def greedy_cycle(path, n, n_all, distances):
    points = np.arange(n_all)
    allowed = np.ones(n_all, dtype='bool')

    allowed[path] = False

    for i in range(len(path), n):
        points_to_check = points[allowed]

        path_extensions = greedy.find_path_extensions(path, points_to_check, distances)
        position, point_idx = np.unravel_index(path_extensions.argmin(), path_extensions.shape)

        path.insert(position, points_to_check[point_idx])
        allowed[points_to_check[point_idx]] = False

    return path
