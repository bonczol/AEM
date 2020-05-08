import numpy as np
import utilites as ut
import local_search_algorithms as ls


AVG_MSLS_SCORE = 15433.7


def multiple_start_ls(distances):
	m = 100
	best_path = ls.steepest_edges(distances)
	best_score = ut.evaluate(best_path, distances)

	print("best: ", best_score)

	for i in range(m-1):
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

	while True:
		path = steepest(distances, path, outside, swap_actions, exchange_actions)
		score = ut.evaluate(path, distances)

		if score < AVG_MSLS_SCORE:
			break
		else:
			path = perturbation_small(path)

	return path


def perturbation_small(path):
	max_offset = int(path.shape[0] / 4)

	p1 = np.random.randint(max_offset)
	p2 = p1 + 1 + np.random.randint(max_offset)
	p3 = p2 + 1 + np.random.randint(max_offset)
	return np.concatenate((path[:p1], path[p3:], path[p2:p3], path[p1:p2]))


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
