import numpy as np
import itertools
import sys
np.set_printoptions(threshold=sys.maxsize)


def steepest(distances):
	n_all = distances.shape[0]
	n = int(np.ceil(n_all / 2))

	path, outside = random_path(n, n_all)

	# Generate actions
	swap_actions = swap_vertices_actions(n)
	exchange_actions = exchange_vertices_actions(path.shape[0], outside.shape[0])


def random_path(length, dataset_size):
	points = np.arange(dataset_size)
	np.random.shuffle(points)
	path, outside = points[:length], points[length:]
	return path, outside


def swap_vertices(path, v1, v2):
	path[v1], path[v2] = path[v2], path[v1]


def exchange_vertices(path, vp, outside, vo):
	path[vp], outside[vo] = path[vo], outside[vp]


def swap_vertices_actions(path_length):
	return np.array(list(itertools.combinations(np.arange(path_length), 2)), "int64")


def exchange_vertices_actions(path_length, outside_length):
	product = itertools.product(np.arange(path_length), np.arange(outside_length))
	return np.array(list(product), "int64")


