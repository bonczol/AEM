import numpy as np
import utilites as ut


def nearest_neighbor(start_point, distances):
	n_all = distances.shape[0]
	n = int(np.ceil(n_all / 2))

	points = np.arange(n_all)

	# Add start point to path
	path = np.zeros(n, dtype='int64')
	path[0] = start_point				

	# Set nearest point as visited
	allowed = np.ones(n_all, dtype='bool')
	allowed[start_point] = False

	for i in range(n-1):
		last_point = path[i]
		nearest_point = find_nearest_neighbour(last_point, points, allowed, distances)
		path[i+1] = nearest_point
		allowed[nearest_point] = False

	return path


def find_nearest_n_points(point, distances, n):
	return np.argsort(distances[point])[1:n+1]


def find_nearest_neighbour(point, points, allowed, distances):
	return points[allowed][np.argmin(distances[point, allowed])]


def greedy_cycle(start_point, distances):  # dodawanie kolenych łuków do cyklu Hamiltona
	n_all = distances.shape[0]
	n = int(np.ceil(n_all / 2))

	points = np.arange(n_all)
	allowed = np.ones(n_all, dtype='bool')

	# Add start point and set as visted
	path = [start_point]
	allowed[start_point] = False

	# Add start point's nearest neighbour and set as visited
	nearest_point = find_nearest_neighbour(start_point, points, allowed, distances)
	path.append(nearest_point)
	allowed[nearest_point] = False

	for i in range(2, n):
		points_to_check = points[allowed]

		# Create matrix with path length after all posible insertions and pick best cell
		path_extensions = find_path_extension(path, points_to_check, distances)
		position, point_idx = np.unravel_index(path_extensions.argmin(), path_extensions.shape)

		path.insert(position, points_to_check[point_idx])
		allowed[points_to_check[point_idx]] = False

	return path


def find_path_extension(path, points, distances):
	ext = [[evaluate(path.copy(), point, position, distances) for point in points] for position in range(len(path))]
	return np.array(ext)


def evaluate(path, point, position, distances):
	path.insert(position, point)
	return ut.evaluate(path, distances)