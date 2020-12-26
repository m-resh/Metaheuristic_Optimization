import numpy as np
from NP_hard import traveling_salesman
import time


def geneticAlgorithm(num_cities=50, num_iter=1000, max_count=100, pop_size=1000, selection_size=900):

    ts = traveling_salesman(num_cities)
    counter = 0
    Pr = 0.5
    best_len = 1e+7
    best_tour = 'no tour found!'
    population = np.zeros((pop_size, 6))
    t0 = time.time()

    ### initializing a random population
    for p in range(pop_size):
        population[p] = np.random.choice(num_cities, size=6, replace=False)

    for i in range(num_iter):
        ### fitness (defined in a way to be used as probablities for selection, so larger value is better)
        lengths = np.zeros((pop_size))
        for p in range(pop_size):
            ts.new_tour(population[p])
            lengths[p] = ts.tour_length()
        sorted_indices = np.argsort(lengths)
        population = population[sorted_indices]
        lengths = lengths[sorted_indices]
        fitness = (-lengths + np.max(lengths) + 1)  # it is always greater than zero
        fitness = fitness / np.sum(fitness)  # so that every value is between 0 and 1 and the sum of all is 1

        ### selection
        selected_indices = np.random.choice(pop_size, size=selection_size, replace=True, p=fitness)

        ### mutation
        parents = population[selected_indices]
        mutated = np.zeros_like(parents)
        for s in range(selection_size):
            mutated[s] = parents[s]
            rnd = np.random.randint(6)  # choosing a random element for mutation
            options = list(set(range(num_cities))-set(mutated[s]))
            random_new_element = np.random.choice(options)
            ## we want mutation to happen with a probability, not always. so, we will choose between the old value and
            ## the new one according to that probability.
            mutated[s, rnd] = np.random.choice(np.append(mutated[s, rnd], random_new_element), p=[1-Pr, Pr])

        ### cross over
        parents = mutated
        children = np.zeros_like(parents)
        for c in range(np.int(selection_size/2)):
            children[2*c] = np.append(parents[2*c, :3], parents[2*c+1, -3:])
            children[2*c+1] = np.append(parents[2*c+1, :3], parents[2*c, -3:])
        ### after this we might have tours with repetitive cities, let's delete those
        obj = np.array([])
        for c in range(children.shape[0]):
            cities = np.unique(children[c])
            if(len(cities) != 6):
                obj = np.append(obj, c)
        obj = obj.astype(int)
        children = np.delete(children, obj, axis=0)

        ### substituting the new children with the worst part of the population
        population[-children.shape[0]:] = children

        best_new_len = lengths[0]
        if(best_new_len < best_len):
            counter = 0
            best_len = best_new_len
        else:
            counter += 1
        if(counter >= max_count):
            break

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
    geneticAlgorithm(num_cities=50, num_iter=1000, max_count=100, pop_size=1000, selection_size=900)
