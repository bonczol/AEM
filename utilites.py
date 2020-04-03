import numpy as np
import matplotlib.pyplot as plt


def load(filename):
	return np.genfromtxt(filename, skip_header=6, skip_footer=1, dtype='int64')[:,1:] # Skip numeration


def calc_distance_matrix(points):
	n = points.shape[0]
	distances = np.zeros((n,n), dtype='int64')
	for i, point in enumerate(points):
		distances[i] = np.round(np.sqrt(np.power(points[:,0] - point[0], 2) + np.power(points[:,1] - point[1], 2)))

	return distances


def print_plot(data, start_point, points, title):
	plt.plot((data[points[0], 0],data[points[-1], 0]), (data[points[0], 1],data[points[-1], 1]), "o:", color="green",  linewidth=2, alpha=0.5)
	plt.plot(data[points, 0], data[points, 1], "+:", color="green", linewidth=2, alpha=0.5)
	plt.plot(data[:, 0], data[:, 1], "o:", linewidth=0, alpha=0.5)
	# plt.plot(data[start_point, 0], data[start_point, 1], "ro:", linewidth=0)
	plt.xlabel("lx")
	plt.ylabel("ly")
	plt.title(title)
	plt.grid(True)
	plt.show()


def evaluate(path, distances):
	length = 0
	for i in range(len(path)-1):
		length += distances[path[i], path[i+1]]

	# Back to first point
	length += distances[path[-1], path[0]]
	return length




