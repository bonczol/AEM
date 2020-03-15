import numpy as np


def nearest_neighbor(start_point, distances):
	n_all = distances.shape[0]
	n = int(np.ceil(n_all / 2))

	# Add start point to path
	path = np.zeros(n, dtype='int64')
	path[0] = start_point				

	# Find n nearest point around start point
	nearest_points = find_nearest_n_points(start_point, distances, n)
	points = np.arange(n_all)

	# Set nearest points as allowed to take part in searching
	allowed = np.zeros(n_all, dtype='bool')
	for point in nearest_points:
		allowed[point] = True

	for i in range(0, len(nearest_points)):
		last_point = path[i]
		nearest_point = points[allowed][np.argmin(distances[last_point, allowed])]
		path[i+1] = nearest_point
		allowed[nearest_point] = False

	return path


def find_nearest_n_points(point, distances, n):
	return np.argsort(distances[point])[1:n]
