from local_search_algorithms import *
import numpy

# Candidates

def swap_candidate_edges_actions(path, distances, k, n):
	actions = set()

	for i in range(n):
		# Skip actions like: (n,n), (n, n_prev), (n, n_next)
		forbidden_idxs = np.array([(i - 1) % n, i, (i + 1) % n])

		# Sort path indexes by distance
		sorted_idxs = np.unravel_index(np.argsort(distances[i][path], axis=None), distances[i][path].shape)[0]

		# Remove forbbiden points and get 'k' nearest
		knn_idxs = sorted_idxs[~np.in1d(sorted_idxs, forbidden_idxs)][:k]

		# Sort tuples and add to set(unique)
		new_actions = [(t[0], t[1]) if t[0] < t[1] else (t[1], t[0]) for t in itertools.product([i], knn_idxs)]
		actions.update(new_actions)

	return np.array(list(actions))


def steepest_candidates(distances):
	# Only 'k' nearest vertices are candidates
	k = 5 

	n_all = distances.shape[0]
	n = int(np.ceil(n_all / 2))

	path, outside = random_path(n, n_all)

	swap_actions = swap_candidate_edges_actions(path, distances, k, n)
	exchange_actions = exchange_vertices_actions(path.shape[0], outside.shape[0])

	while True: 
		swap_delta = [calc_swap_edges_delta(s, path, distances) for s in swap_actions]
		exchange_delta = [calc_exchange_delta(e, path, outside, distances) for e in exchange_actions]

		smax_idx = np.argmax(swap_delta)
		emax_idx = np.argmax(exchange_delta)

		smax = swap_delta[smax_idx]
		emax = exchange_delta[emax_idx]

		# print(swap_actions[smax_idx], exchange_actions[emax_idx])

		if smax <= 0 and emax <= 0:
			break

		if smax > emax:
			swap_edges(path, swap_actions[smax_idx])
		else:
			exchange_vertices(path, outside, exchange_actions[emax_idx])
			swap_actions = swap_candidate_edges_actions(path, distances, k, n)

	return path

