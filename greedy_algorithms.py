import numpy as np
import utilites as ut
import numpy.ma as ma


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


def greedy_cycle(start_point, distances):  # dodawanie kolenych łuków do cyklu Hamiltona
	n_all = distances.shape[0]
	n = int(np.ceil(n_all / 2))

	points = np.arange(n_all)
	allowed = np.ones(n_all, dtype='bool')

	# Add start point and set as visted
	path = [start_point]
	allowed[start_point] = False

	for i in range(1, n):
		points_to_check = points[allowed]

		# Create matrix with path length after all posible insertions and pick best cell
		path_extensions = find_path_extensions(path, points_to_check, distances)
		position, point_idx = np.unravel_index(path_extensions.argmin(), path_extensions.shape)

		path.insert(position, points_to_check[point_idx])
		allowed[points_to_check[point_idx]] = False

	return path


def greedy_cycle_with_regret(start_point, distances):  # dodawanie kolenych łuków do cyklu Hamiltona
	n_all = distances.shape[0]
	n = int(np.ceil(n_all / 2))

	points = np.arange(n_all)
	allowed = np.ones(n_all, dtype='bool')

	# Add start point and set as visted
	path = [start_point]
	allowed[start_point] = False

	nearest_point = find_nearest_neighbour(start_point, points, allowed, distances)
	path.append(nearest_point)
	allowed[nearest_point] = False

	while n > len(path):
		points_to_check = points[allowed]

		# Create matrix with path length after all posible insertions and pick best cell
		path_extensions = find_path_extensions(path, points_to_check, distances)
		insert_position, insert_point_idx = np.unravel_index(path_extensions.argmin(), path_extensions.shape)
		insert_point = points_to_check[insert_point_idx]

		path_cuts = find_path_reductions(path, [(insert_position-1) % len(path), insert_position%len(path)], distances)
		del_point_idx = np.argmax(path_cuts)

		if path_extensions[insert_position, insert_point_idx] >= path_cuts[del_point_idx]:
			path.insert(insert_position, insert_point)
			allowed[insert_point] = False
		else:
			allowed[path.pop(del_point_idx)] = True
			bias = -1 if del_point_idx < insert_position else 0
			
			path.insert(insert_position+bias, insert_point)
			allowed[insert_point] = False

	return path


def find_nearest_n_points(point, distances, n):
	return np.argsort(distances[point])[1:n+1]


def find_nearest_neighbour(point, points, allowed, distances):
	return points[allowed][np.argmin(distances[point, allowed])]


def find_path_extensions(path, points, distances):
	return np.array([[length_extension(path, position, point, distances) for point in points] for position in range(len(path))])


def find_path_reductions(path, forbidden, distances):
	ext = [-1 if position in forbidden else length_reduction(path.copy(), position, distances) for position in range(len(path))]
	return np.array(ext)


def evaluate_cut(path, position, distances):
	path.pop(position)
	return ut.evaluate(path, distances)


def evaluate_add(path, position, point, distances):
	path.insert(position, point)
	return ut.evaluate(path, distances)


def length_extension(path, position, point, distances):
	i = path[position-1]
	j = path[(position) % len(path)]
	i_point = distances[i, point]
	j_point = distances[j, point]
	i_j = distances[i,j]
	return i_point + j_point - i_j


def length_reduction(path, position, distances):
	point = path[position]
	i = path[position-1]
	j = path[(position+1) % len(path)]
	i_point = distances[i, point]
	j_point = distances[j, point]
	i_j = distances[i, j]
	return i_point + j_point - i_j
