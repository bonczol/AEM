import local_search_algorithms as ls
import utilites as ut
import numpy as np

import matplotlib.pyplot as plt
from scipy import stats


def global_convex(filename, distances, f):
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
        mean_vertices_similar = 0
        mean_edges_similar = 0
        for another_path in optimums:
            mean_vertices_similar += vertices_similarity(path, another_path)
            mean_edges_similar += edges_similarity(path, another_path)
        mean_vertices_similarities[i] = mean_vertices_similar/optimums.shape[0]
        mean_edges_similarities[i] = mean_edges_similar/optimums.shape[0]

    print(f, stats.pearsonr(score, vertices_similarities)[0])
    print(f, stats.pearsonr(score, edges_similarities)[0])
    print(f, stats.pearsonr(score, mean_vertices_similarities)[0])
    print(f, stats.pearsonr(score, mean_edges_similarities)[0])
    print("etap")

    plt.plot(score,edges_similarities, "o:", color="green", linewidth=0, alpha=0.2)
    # plt.plot(score, vertices_similarities, "o:", color="green", linewidth=0, alpha=0.2)
    plt.plot(score[best_path_index], edges_similarities[best_path_index], "o:", linewidth=0, alpha=0.5)
    plt.xlabel("lx")
    plt.ylabel("ly")
    title = "edges_similarities - "+f
    plt.title(title)
    plt.savefig(f"plots/{title}.png")
    plt.grid(True)
    # plt.show()

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