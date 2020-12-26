import numpy as np
import matplotlib.pyplot as plt
from NP_hard import traveling_salesman
from Random_Search import randomSearch
from Brute_Force import bruteForce
from Genetic_Algorithm import geneticAlgorithm
from Differential_Evolution import differentialEvolution


salesman = traveling_salesman(num_cities=50)  # initialize the TS's world

# create a random tour for the TS, plot and calculate length of tour
random_initial_tour = salesman.random_tour(num_stops=6)
salesman.plot()
print("length of the tour: ", salesman.tour_length())

# assign a new tour to the TS, plot and calculate length
salesman.new_tour(tour=np.arange(6).tolist())
salesman.plot()
print("length of the tour: ", salesman.tour_length())

# assign a new tour to the TS, plot and calculate length
salesman.new_tour(tour=[1, 6, 7, 28, 42, 9])
salesman.plot()
print("length of the tour: ", salesman.tour_length())


### comparing the 4 optimization methods:
# plots are integrated in the algorithms

randomSearch(num_cities=50, num_iter=1000)

bruteForce(num_cities=50)

geneticAlgorithm(num_cities=50, num_iter=1000, max_count=100, pop_size=1000, selection_size=900)

differentialEvolution(num_cities=50, max_iter=1200, max_count=400, pop_size=200)
