import numpy as np
import utilites as ut


def regretdef(start_point, distances):  # dodawanie kolenych łuków do cyklu Hamiltona
    n_all = distances.shape[0]
    n = int(np.ceil(n_all / 2))

    points = np.arange(n_all)
    allowed = np.zeros(n_all, dtype='bool')

    # Add start point and set as visted
    path = [start_point]
    allowed[start_point] = False
    nearest_point = find_nearest_n_points(start_point, distances, n)
    allowed[nearest_point] = True

    for i in range(1, n):
        points_to_check = points[allowed]

        # Create matrix with path length after all posible insertions and pick best cell
        greedy_path_extensions = find_path_extension(path, points_to_check, distances)
        position, point_idx = np.unravel_index(greedy_path_extensions.argmin(), greedy_path_extensions.shape)

        greedy_path=path.copy()
        greedy_position=position
        greedy_point=points_to_check[point_idx]
        greedy_path.insert(greedy_position, greedy_point)

        path_extensions = find_path_extension(greedy_path, points_to_check, distances)
        path_extensions[:,point_idx]=0
        a = path_extensions.min(axis=0)
        b = greedy_path_extensions.min(axis=0)
        regret= a -b

        point_idx =regret.argmax()
        position=np.argmin(greedy_path_extensions[:,point_idx])
        path.insert(position, points_to_check[point_idx])
        allowed[points_to_check[point_idx]] = False

    return path


def find_nearest_n_points(point, distances, n):
	return np.argsort(distances[point])[1:n+1]


def find_nearest_neighbour(point, points, allowed, distances):
    return points[allowed][np.argmin(distances[point, allowed])]


def find_path_extension(path, points, distances):
    ext = [[evaluate(path.copy(), point, position, distances) for point in points] for position in range(len(path))]
    return np.array(ext)


def evaluate(path, point, position, distances):
    path.insert(position, point)
    return ut.evaluate(path, distances)

def find_nearest_n_points(point, distances, n):
    return np.argsort(distances[point])[1:n]
