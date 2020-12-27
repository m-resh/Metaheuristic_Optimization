import numpy as np
import matplotlib.pyplot as plt
from NP_hard import traveling_salesman
import time


def randomSearch(num_cities=50, num_iter=1000):

    ts = traveling_salesman(num_cities)
    best_len = 1e+7
    counter = 0
    t0 = time.time()

    for i in range(num_iter):
        tour = np.random.choice(num_cities, size=6, replace=False)
        ts.new_tour(tour)
        length = ts.tour_length()
        for j in range(100):
            new_tour = np.copy(tour)
            opt_list = list(set(range(num_cities))-set(new_tour))
            new_random_city = np.random.choice(opt_list)
            new_tour[np.random.randint(6)] = new_random_city  # change one of the cities randomly, with the new random city
            ts.new_tour(new_tour)
            if (ts.tour_length() < length):
                length = ts.tour_length()
                tour = new_tour
        if(length < best_len):
            counter = 0
            best_len = length
            best_tour = tour
        else:
            counter += 1
        if(counter >= 200):
            break

    t1 = time.time()
    ts.new_tour(best_tour)
    print('wall time:', np.around(t1-t0, 3), '(s)', 'number of iterations:', i)
    print('iteration time:', np.around((t1-t0)/i, 3), '(s)')
    print('best tour: ', best_tour, 'with length: ', best_len)
    ts.plot()
    return best_tour


if __name__ == "__main__":
    randomSearch(num_cities=50, num_iter=1000)
