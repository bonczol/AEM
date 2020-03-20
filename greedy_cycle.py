import numpy as np
import utilites as ut


def greedy_cycle(start_point, distances):  # dodawanie kolenych łuków do cyklu Haminga
    n_all = distances.shape[0]
    n = int(np.ceil(n_all / 2))

    # Find n nearest point around start point
    points = np.arange(n_all)

    # Set nearest points as allowed to take part in searching
    allowed = np.zeros(n_all, dtype='bool')
    allowed[:] = True
    allowed[start_point] = False

    # Tworzenie podstawowego cyklu
    last_point = points[allowed][np.argmin(distances[start_point, allowed])]
    graf_wyj = [start_point, last_point]
    allowed[last_point] = False

    point = start_point
    position = 0
    # Rozciąganie cyklu ( założyłem, że najkrótszy łuk jest do najbliżeszego wierzchołka)
    # jak na razie nearest_neighbor ma i tak mniejszą długość
    for i in range(n - 2):
        # TODO poprawic wyszukiwanie najblizszego wierzchołka
        global_min = np.inf
        for counter, j in enumerate(graf_wyj):
            local_min = np.argmin(distances[j, allowed])
            if local_min < global_min:
                global_min = local_min
                point = points[allowed][local_min]
                position = counter
        nearest_point = point
        allowed[nearest_point] = False

        # TODO poprawic wstawianie
        if graf_wyj.__len__() > position + 1:

            if distances[nearest_point, graf_wyj[position + 1]] < distances[nearest_point, graf_wyj[position - 1]]:
                graf_wyj.insert(position + 1, point)
                print(11)
            else:
                graf_wyj.insert(position, point)
                print(12)

        else:
            if distances[nearest_point, graf_wyj[-1]] < distances[nearest_point, graf_wyj[position - 1]]:
                graf_wyj.insert(position + 1, point)
                print(21)
            else:
                graf_wyj.insert(position, point)
                print(22)
    print(graf_wyj)

    return graf_wyj


def greedy_cycle_v2(start_point, distances):  # dodawanie kolenych łuków do cyklu Haminga
    n_all = distances.shape[0]
    n = int(np.ceil(n_all / 2))

    points_to_check = [x for x in range(n_all)]
    print(points_to_check)
    print(start_point)
    points_to_check.remove(start_point)

    global_solution = [start_point]

    for i in range(n - 1):
        global_result = np.inf
        local_solution = global_solution.copy()
        for count, j in enumerate(global_solution):
            for k in points_to_check:
                temp_solution = global_solution.copy()
                temp_solution.insert(count, k)
                local_result = ut.evaluate(temp_solution, distances)
                if local_result < global_result:
                    last_point=k
                    global_result = local_result
                    local_solution = temp_solution.copy()
        points_to_check.remove(last_point)
        global_solution = local_solution.copy()
    print(global_solution)

    return global_solution
