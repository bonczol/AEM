import utilites as ut
import numpy as np
import local_search_based_algorithms as alg
import greedy_algorithms as ga
import time

def test(algorithm, instance, distances):
	n = 10
	solutions = []
	times = []
	results = np.zeros(n, dtype="int64")

	for i in range(n):
		start = time.perf_counter()
		solutions.append(algorithm(distances))
		end = time.perf_counter()
		times.append(end - start)
		results[i] = ut.evaluate(solutions[i], distances)
		print(results[i], times[i])

	best_start_point = np.argmin(results)
	best_solution = solutions[best_start_point]
	min_val = np.min(results)
	max_val = np.max(results)
	avg_val = np.mean(results)
	avg_time = np.mean(times)

	return best_solution, best_start_point, min_val, max_val, avg_val, avg_time


def main():
	start_point = 0
	instances_names = ["kroA100.tsp","kroB100.tsp", "kroA200.tsp", "kroB200.tsp"]
	instance = ut.load(f'instances/{instances_names[3]}')
	distances = ut.calc_distance_matrix(instance)

	best_solution, best_start_point, min_val, max_val, avg_val, avg_time = test(alg.multiple_start_ls, instance, distances)
	print(min_val, max_val, avg_val, avg_time)
	ut.print_plot(instance, 0, best_solution, "LS")
  
if __name__== "__main__":
  main()