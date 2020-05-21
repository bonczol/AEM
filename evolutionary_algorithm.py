import time
import random
import numpy as np
import utilites as ut
import local_search_algorithms as ls
import local_search_based_algorithms as lsb

AVG_MSLS_TIME = 30


def evolutionary(distances):
    start = time.perf_counter()
    ls_count = 1

    n_all = distances.shape[0]
    n = int(np.ceil(n_all / 2))

    swap_actions = ls.swap_edges_actions(n)
    exchange_actions = ls.exchange_vertices_actions(n, n_all - n)

    # Individual in population is a tuple (path, score)
    population_size = 20
    population = generate_population(population_size, n, n_all)
    scores = np.array([ut.evaluate(path, distances) for path in population])
    worst_idx = np.argmax(scores)

    while time.perf_counter() - start < AVG_MSLS_TIME:
        path1, path2 = pick_two_random(population)
        recombined_path = recombination(path1, path2)

        new_path, _ = lsb.steepest(distances, recombined_path, get_outside(recombined_path, n_all), swap_actions, exchange_actions)
        new_score = ut.evaluate(new_path, distances)
        ls_count = ls_count + 1

        if new_score < scores[worst_idx] and is_unique_in_population(population, new_path):
            population[worst_idx] = new_path
            scores[worst_idx] = new_score
            worst_idx = np.argmax(scores)

    return population[np.argmin(scores)], ls_count


# TODO zastanwoić się czy generujemy populację losowo czy np. z greedy cycle albo nawet część losowo i częśc greedy
def generate_population(population_size, n, n_all):
    population = np.zeros((population_size, n), int)
    current_population_size = 0

    while current_population_size < population_size:
        path, _ = ls.random_path(n, n_all)
        if is_unique_in_population(population, path):
            population[current_population_size] = path
            current_population_size += 1

    return population


def pick_two_random(population):
    picked_keys = np.random.choice(np.arange(population.shape[0]), 2, replace=False)
    return population[picked_keys[0]], population[picked_keys[1]]


def recombination(path, another_path):
    # TODO zrobić rekombinacje na dwóch ścieżkach
    return np.arange(0, 100)


def get_outside(path, n_all):
    return np.setdiff1d(np.arange(n_all), path)


def paths_equal(path, another_path):
    doubled_path = np.concatenate((path, path))

    for i in range(doubled_path.shape[0]):
        if np.array_equal(another_path, doubled_path[i: i + doubled_path.shape[0]]):
            return True
    
    return False
        
        
def is_unique_in_population(population, path):
    for individual in population:
        if paths_equal(individual, path):
            return False
    
    return True
    

