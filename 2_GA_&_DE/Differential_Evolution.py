import numpy as np
from NP_hard import traveling_salesman
import time


def differentialEvolution(num_cities=50, max_iter=1200, max_count=400, pop_size=200):

    ts = traveling_salesman(num_cities)
    counter = 0
    F = 0.5
    Pr = 0.3
    best_len = 1e+7
    best_tour = 'no tour found!'
    population = np.zeros((pop_size, 6))
    new_population = np.zeros((pop_size, 6))
    lengths = np.zeros((pop_size))
    t0 = time.time()

    ### initializing a random population
    for p in range(pop_size):
        population[p] = np.random.choice(num_cities, size=6, replace=False)

    # so that we wouldn't update the current population until the end of each iteration
    new_population = np.copy(population)

    for i in range(max_iter):
        for p in range(pop_size):
            gi1 = population[p]
            ts.new_tour(gi1)
            ### fitness
            lengths[p] = ts.tour_length()
            ### mutation
            options = list(set(range(num_cities))-set([p]))
            indices = np.random.choice(options, size=2, replace=False)  # choosing gi1, gi2 & gi3
            [gi2, gi3] = population[indices]
            gi = np.array(np.remainder(np.around(gi1 + F*(gi2 + gi3)), num_cities), dtype=int)
            ### crossover
            g_prime = np.zeros_like(gi)
            for b in range(6):
                g_prime[b] = np.random.choice(np.append(gi1[b], gi[b]), p=[1-Pr, Pr])
            ## if we have repetitive cities, we will not proceed
            cities = np.unique(g_prime)
            if(len(cities) == 6):
                ts.new_tour(g_prime)
                if(ts.tour_length() < lengths[p]):
                    new_population[p] = g_prime
                    lengths[p] = ts.tour_length()
        population = np.copy(new_population)
        sorted_indices = np.argsort(lengths)
        best_new_len = lengths[sorted_indices[0]]
        if(best_new_len < best_len):
            counter = 0
            best_len = best_new_len
        else:
            counter += 1
        if(counter >= max_count):
            break

    sorted_indices = np.argsort(lengths)
    population = population[sorted_indices]
    best_tour = population[0]
    ts.new_tour(best_tour)
    best_len = ts.tour_length()
    t1 = time.time()
    print('wall time:', np.around(t1-t0, 3), '(s)', 'number of iterations:', i)
    print('iteration time:', np.around((t1-t0)/i, 3), '(s)')
    print('best tour: ', best_tour, 'with length: ', best_len)
    ts.plot()
    return best_tour


if __name__ == "__main__":
    differentialEvolution(num_cities=50, max_iter=1200, max_count=400, pop_size=200)
