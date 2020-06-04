import utilites as ut
import numpy as np
import evolutionary_algorithm as alg
import greedy_algorithms as ga
import time
import global_convex

def test(algorithm, instance, distances):
	n = 10
	solutions = []
	times = []
	ls_counts = []
	results = np.zeros(n, dtype="int64")

	for i in range(n):
		start = time.perf_counter()
		solution, count = algorithm(distances)
		solutions.append(solution)
		ls_counts.append(count)
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
	min_ls_count = np.min(ls_counts)
	max_ls_count = np.max(ls_counts)
	avg_ls_count = np.mean(ls_counts)

	return best_solution, best_start_point, min_val, max_val, avg_val, avg_time, min_ls_count, max_ls_count, avg_ls_count


def main():
	N = 1000
	instances_names = ["kroA100.tsp", "kroB100.tsp", "kroA200.tsp", "kroB200.tsp"]
	opt_filenames = ["kroA100_opt.txt", "kroB100_opt.txt", "kroA200_opt.txt", "kroB200_opt.txt"]

	for instance_name, filename in zip(instances_names, opt_filenames):
		instance = ut.load(f'instances/{instance_name}')
		distances = ut.calc_distance_matrix(instance)
		# global_convex.generate_optimums(f'optimums/{filename}', N, distances)
		global_convex.global_convex(f'optimums/{filename}', distances)

	# best_solution, best_start_point, min_val, max_val, avg_val, avg_time, min_ls_count, max_ls_count, avg_ls_count = test(alg.evolutionary, instance, distances)
	# print("Results: ", min_val, max_val, avg_val, avg_time)
	# print("LS runs: ", min_ls_count, max_ls_count, avg_ls_count)
	# ut.print_plot(instance, 0, best_solution, "Hybrid evo - kroA200")
  
if __name__== "__main__":
  main()