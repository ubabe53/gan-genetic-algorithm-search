from random import uniform, randint, choices, random
from copy import deepcopy
import numpy.random as npr

## Selection
def roulette_wheel(solutions):
    """
    :param solutions:
    :return:
    """
    for x in solutions:
        if x.fitness < 0:
            x.set_fitness(0)

    max = sum([c.fitness for c in solutions])
    selection_probs = [(c.fitness / max) for c in solutions]
    indexes = npr.choice(len(solutions),2, p=selection_probs)
    return solutions[indexes[0]],solutions[indexes[1]]

## Crossover
def singlepoint_crossover(solution1, solution2):
    point = randint(0, len(solution1.representation)-1)

    offspring1 = deepcopy(solution1)
    offspring2 = deepcopy(solution2)

    for i in range(point, len(solution2.representation)):
        offspring1.representation[i] = solution2.representation[i]
        offspring2.representation[i] = solution1.representation[i]

    return offspring1, offspring2


## Mutation
def uniform_mutation(solution):
    """
    :param solution: A list representing the individual
    :return: The individual after mutation
    """
    mutation_rate = 2/len(solution.representation)

    for i, gene in enumerate(solution.representation):
        chance = random()
        if chance < mutation_rate:
            random_value = randint(-1, 1)
            gene = gene + random_value
            if gene == 7:
                gene = gene - 2
            if gene == 0:
                gene = gene + 2
            solution.representation[i] = gene
    return solution

## Replacement

def elitism_replacement(current_population, new_population):
    """
    :param current_population: The current population
    :param new_population: The new population
    :return: The new population with the fittest ones
    """

    if current_population.fittest.fitness > new_population.fittest.fitness:
        new_population.solutions[0] = current_population.solutions[-1]
    elif current_population.fittest.fitness > new_population.solutions[-2].fitness:
        new_population.solutions[1] = current_population.solutions[-1]
    if current_population.solutions[-2].fitness > new_population.solutions[-2].fitness:
        new_population.solutions[2] = current_population.solutions[-2]

    return deepcopy(new_population)
