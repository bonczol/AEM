import numpy as np
import utilites as ut
from local_search_algorithms import steepest_edges


def multiple_start_ls(distances):
	m = 10
	best_path = steepest_edges(distances)
	best_score = ut.evaluate(best_path, distances)

	for i in range(m-1):
		result_path = steepest_edges(distances)
		result_score = ut.evaluate(result_path, distances)

		if result_score < best_score:
			best_path = result_path
			best_score = result_score

	return best_path


def iterated_ls1(distances, avg_score_msls):
	path = steepest_edges(distances)

	while True:
		path = perturbation_xs(path)
		path = steepest_edges(path)
		score = ut.evaluate(path)

		if score < avg_score_msls:
			break

	return path


def perturbation_xs(path):
	return np.array(10)









