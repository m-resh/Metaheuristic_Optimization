# Metaheuristic_Optimization
This repo consists of 3 parts. In the first part, classical optimazion methods, e.g. random search, gradient descend, and newton-raphson, are implemented and compared on 4 general mathematical functions. Figures 1-4 illustrate the jurney of the gradient descend algorithm, as an example, on the surfaces of the 4 proposed functions.

<img src="figures/1/gd0.png" alt="drawing" width="400"/> | <img src="figures/1/gd1.png" alt="drawing" width="400"/>
:--:|:--:
_Figure 1: Problem 0_ | _Figure 2: Problem 1_

<img src="figures/1/gd2.png" alt="drawing" width="400"/> | <img src="figures/1/gd3.png" alt="drawing" width="400"/>
:--:|:--:
_Figure 3: Problem 2_ | _Figure 4: Problem 3_



In the next part, Genetic Algorithm, Differential Evolution, Brute-force, and random search are implemented and compared on the traveling salesmen problem, as an example of NP-hard problems. In this problem, we are looking for a tour through 6 cities with the least cost (i.e. least total distance travelled). Figure 5 is a visualizations of this problem which shows the optimal tour for the chosen random seed.

| ![](figures/2/optimal.png) |
|:--:|
| _Figure 5: The traveling salesman problem_ |

Finally, in part 3, our optimization objectives are blackbox problems! With no knowledge of how these functions work, Particle Swarm Optimization is implemented and compared alongside genetic algorithm, grid search, and random search.
