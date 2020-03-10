import numpy as np
import matplotlib.pyplot as plt

def load(filename):
	return np.genfromtxt(filename, skip_header=6, skip_footer=1, dtype='int64')[:,1:] # Skip enumeration

def calc_distance_matrix(points):
	n = points.shape[0]
	distances = np.zeros((n,n), dtype='int64')
	for i, point in enumerate(points):
		distances[i] = np.round(np.sqrt(np.power(points[:,0] - point[0], 2) + np.power(points[:,1] - point[1], 2)))

	return distances

def find_greedy(tab):
	steps=50
	c=30 			#start point
	points=[c]
	for i in range(steps):    #tutaj jest taki problem że z tablicy trzeba byłoby usuwać wężły w którcyh się już było
		tab[c, c] = 9999
		d = np.argmin(tab[c,:])
		tab[c,d]=tab[d,c]=9999
		c=d
		points.append(c)
	return points

def print_plot(data, points):
	plt.plot(data[:, 0], data[:, 1], "o:", linewidth=0, alpha=0.5)
	plt.plot(data[points, 0], data[points, 1], "+:", color="green", linewidth=2, alpha=0.5)
	# plt.legend("Dane x, y\nPrzemieszczenie: ", loc="upper left")
	plt.xlabel("lx")
	plt.ylabel("ly")
	plt.title("AEM")
	plt.grid(True)
	plt.show()

instance = load("instances/kroA100.tsp")
distances = calc_distance_matrix(instance)
points=find_greedy(distances)
print_plot(instance, points)

# print(distances)