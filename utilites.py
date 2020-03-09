import numpy as np


def load(filename):
	return np.genfromtxt(filename, skip_header=6, skip_footer=1, dtype='int64')[:,1:] # Skip enumeration

def calc_distance_matrix(points):
	n = points.shape[0]
	distances = np.zeros((n,n), dtype='int64')
	
	for i, point in enumerate(points):
		distances[i] = np.round(np.sqrt(np.power(points[:,0] - point[0], 2) + np.power(points[:,1] - point[1], 2)))

	return distances


instance = load("instances/kroA100.tsp")
distances = calc_distance_matrix(instance)

print(distances)