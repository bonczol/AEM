import local_search_algorithms as ls
import utilites as ut
import numpy as np

import matplotlib.pyplot as plt
from scipy import stats


def global_convex(filename, distances):
    optimums = np.loadtxt(filename, dtype ='int')

    score = np.empty(optimums.shape[0])
    vertices_similarities = np.empty(optimums.shape[0])
    edges_similarities = np.empty(optimums.shape[0])
    mean_vertices_similarities = np.empty(optimums.shape[0])
    mean_edges_similarities = np.empty(optimums.shape[0])
    correlation = np.empty(optimums.shape[0])

    for i, path in enumerate(optimums):
        score[i] = ut.evaluate(path, distances)

    best_path_index=np.argmin(score)
    best_path = optimums[best_path_index]

    # Similarity to best path
    for i, path in enumerate(optimums):
        vertices_similarities[i] = vertices_similarity(path, best_path)
        edges_similarities[i] = edges_similarity(path, best_path)
        correlation[i] = stats.pearsonr(path, best_path)[0]

    # plt.plot(score, edges_similarities, "o:", color="red", linewidth=0, alpha=0.2)
    # plt.plot(score, vertices_similarities, "o:", color="green", linewidth=0, alpha=0.2)
    plt.plot(score, correlation, "o:", color="green", linewidth=0, alpha=0.2)
    plt.plot(score[best_path_index], vertices_similarities[best_path_index], "o:", linewidth=0, alpha=0.5)
    plt.xlabel("lx")
    plt.ylabel("ly")
    plt.title(filename)
    plt.grid(True)
    plt.show()

    return 0


def generate_optimums(filename, n, distances):
    optimums = np.array([ls.greedy_edges(distances) for i in range(n)])
    np.savetxt(filename, optimums, fmt='%i')


def vertices_similarity(path, another_path):
    common = np.intersect1d(path, another_path)
    return len(common) / len(path)


def edges_similarity(path, another_path):
    common_count=0
    for i in range(path.shape[0]):
        for j in range(another_path.shape[0]):
            if path[i]==another_path[j]:
                if path[i-1]==another_path[j-1] or path[i-1]==another_path[(j+1)% another_path.shape[0]] :
                    common_count += 1
                break
    return common_count / len(path)