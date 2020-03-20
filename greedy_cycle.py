import numpy as np


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
    for i in range(n-2):
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

        #TODO poprawic wstawianie
        if graf_wyj.__len__() > position + 1:

            if distances[nearest_point, graf_wyj[position+1]] < distances[nearest_point, graf_wyj[position - 1]]:
                graf_wyj.insert(position+1, point)
            else:
                graf_wyj.insert(position, point)

        else:
            if distances[nearest_point, graf_wyj[-1]] < distances[nearest_point, graf_wyj[position - 1]]:
                graf_wyj.insert(position+1, point)
            else:
                graf_wyj.insert(position, point)

    print(graf_wyj)

    return graf_wyj
