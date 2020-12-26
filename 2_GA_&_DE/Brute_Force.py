from NP_hard import traveling_salesman
import time
from itertools import permutations, combinations


def bruteForce(num_cities=50):

    ts = traveling_salesman(num_cities)
    best_len = 1e+7
    best_tour = 'no tour found!'
    lst = list(range(num_cities))
    i = 0
    t0 = 1000*time.time()
    for c in combinations(lst, 6):
        lst2 = c[1:]
        for p in permutations(lst2):
            if(p[0] > p[-1]):
                x = [c[0]] + list(p)
                i += 1
                ts.new_tour(x)
                length = ts.tour_length()
                if (length < best_len):
                    best_len = length
                    best_tour = x

    t1 = 1000*time.time()
    print(i, t1-t0, '(ms)')
    print('best tour: ', best_tour, 'with length: ', best_len)
    ts.new_tour(tour=best_tour)
    ts.plot()
    return best_tour


if __name__ == "__main__":
    bruteForce(num_cities=50)
