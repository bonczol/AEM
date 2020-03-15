import utilites as ut
import greedy_algorithms as alg


def main():
	start_point = 0
	instances_names = ["kroA100.tsp","kroB100.tsp"]
	instance = ut.load(f'instances/{instances_names[1]}')
	distances = ut.calc_distance_matrix(instance)
	solution = alg.nearest_neighbor(start_point, distances)
	result = ut.evaluate(solution, distances)

	print(result)
	ut.print_plot(instance, start_point, solution)
  
if __name__== "__main__":
  main()