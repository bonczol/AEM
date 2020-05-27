import local_search_algorithms as ls
import utilites as ut
import numpy as np


def global_convex(filename, distances):
    optimums = np.loadtxt(filename)

    score = np.empty(optimums.shape[0])
    vertices_similarities = np.empty(optimums.shape[0])
    edges_similarities = np.empty(optimums.shape[0])
    mean_vertices_similarities = np.empty(optimums.shape[0])
    mean_edges_similarities = np.empty(optimums.shape[0])

    for i, path in enumerate(optimums):
        score[i] = ut.evaluate(path, distances)

    best_path = optimums[np.argmin(score)]

    # Similarity to best path
    for i, path in enumerate(optimums):
        vertices_similarities[i] = vertices_similarity(path, best_path)
        edges_similarities[i] = edges_similarity(path, best_path)

    # TODO Similarity to all other paths

    return 0


def generate_optimums(filename, n, distances):
    optimums = np.array([ls.greedy_edges(distances) for i in range(n)])
    np.savetxt(filename, optimums, fmt='%i')


def vertices_similarity(path, another_path):
    common = np.intersect1d(path, another_path)
    return len(common) / len(path)


def edges_similarity(path, another_path):
    # TODO
    return 0