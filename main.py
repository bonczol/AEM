import utilites as ut
import greedy_algorithms as alg
import numpy as np


def test(algorithm, instance, distances):
	n = instance.shape[0]
	results = np.zeros(n, dtype="int64")

	for i in range(n):
		solution = algorithm(i, distances)
		results[i] = ut.evaluate(solution, distances)
		print(results[i])

	min_val = np.min(results)
	max_val = np.max(results)
	avg_val = np.mean(results)

	return (min_val, max_val, avg_val)


def main():
	start_point = 0
	instances_names = ["kroA100.tsp","kroB100.tsp"]
	instance = ut.load(f'instances/{instances_names[1]}')
	distances = ut.calc_distance_matrix(instance)
	# results = test(alg.nearest_neighbor, instance, distances)
	# results = test(alg.greedy_cycle, instance, distances)
	results = test(alg.greedy_cycle_with_regret, instance, distances)
	print(results)
  
if __name__== "__main__":
  main()
