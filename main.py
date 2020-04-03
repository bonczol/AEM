import utilites as ut
import numpy as np
import local_search_algorithms as alg


def test(algorithm, instance, distances):
	n = instance.shape[0]
	solutions = []
	results = np.zeros(n, dtype="int64")

	for i in range(n):
		solutions.append(algorithm(i, distances))
		results[i] = ut.evaluate(solutions[i], distances)
		print(results[i])

	best_start_point = np.argmin(results)
	best_solution = solutions[best_start_point]
	min_val = np.min(results)
	max_val = np.max(results)
	avg_val = np.mean(results)

	return best_solution, best_start_point, min_val, max_val, avg_val


def main():
	start_point = 0
	instances_names = ["kroA100.tsp","kroB100.tsp"]
	instance = ut.load(f'instances/{instances_names[1]}')
	distances = ut.calc_distance_matrix(instance)

	#
	# best_solution, best_start_point, min_val, max_val, avg_val = test(alg.greedy_cycle_with_regret, instance, distances)
	#
	# print(min_val, max_val, avg_val)
	# ut.print_plot(instance, best_start_point, best_solution, "Greedy cycle")
	solution = alg.steepest(distances, instance)
	ut.print_plot(instance, 0, solution, "LS")
  
if __name__== "__main__":
  main()