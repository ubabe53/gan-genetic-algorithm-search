from random import uniform, randint, choice, random
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

def adapted_mutation(solution,limits={}):
    """
    :param solution: A list representing the individual
    :return: The individual after mutation
    """
    mutation_rate = 2 / len(solution.representation)

    for i, gene in enumerate(solution.representation):
        chance = random()
        if chance < mutation_rate:
            if i == 0:
                gene = batch_mutation(gene,limits)
                solution.representation[i] = gene
            elif i == 1:
                gene = lr_mutation(gene,limits)
                solution.representation[i] = gene
            elif i in [2, 3]:
                gene = betas_mutation(gene, i, limits)
                solution.representation[i] = gene
            elif i in [4, 5]:
                gene = oneint_mutation(gene, i, limits)
                solution.representation[i] = gene

    return solution


def limits_parser(limits={}):

    batch_limits = [100, 1000]
    if "batch_limits" in limits:
        batch_limits = limits["batch_limits"]

    lr_limits = [0.00005, 0.001]
    if "lr_limits" in limits:
        lr_limits = limits["lr_limits"]

    beta1_limits = [0.50, 0.99]
    if "beta1_limits" in limits:
        beta1_limits = limits["beta1_limits"]

    beta2_limits = [0.700, 0.999]
    if "beta2_lmits" in limits:
        beta2_limits = limits["beta2_limits"]

    n_critic_limits = [2, 10]
    if "n_critic_limits" in limits:
        n_critic_limits = limits["n_critic_limits"]

    weight_gp_limits = [5, 15]
    if "weight_gp_limits" in limits:
        weight_gp_limits = limits["weight_gp_limits"]

    limits_dict = {
        "batch_limits": batch_limits,
        "lr_limits": lr_limits,
        "beta1_limits": beta1_limits,
        "beta2_limits": beta2_limits,
        "n_critic_limits": n_critic_limits,
        "weight_gp_limits": weight_gp_limits
    }

    return limits_dict


def batch_mutation(gene, limits={}):
    """
    :param gene: The gene to mutate
    :param limits: The range of mutation
    :return: The mutated gene
    It takes a gene and add or subtract a random int between 25 and 50, in case it exceed the max or min value
    gets 'pushed' back into the range.
    """
    limits = limits_parser(limits)
    value1 = randint(25,50)
    value2 = -randint(25,50)
    value = choice([value2,value1])
    gene = gene + value
    if gene < limits['batch_limits'][0]:
        gene = limits['batch_limits'][0]
    elif gene > limits['batch_limits'][1]:
        gene = limits['batch_limits'][1]

    return gene


def lr_mutation(gene, limits={}):

    limits = limits_parser(limits)
    value1 = round(npr.uniform(0.00005, 0.00250), 5)
    value2 = -round(npr.uniform(0.00005, 0.00250), 5)
    value = choice([value2, value1])
    gene = round(gene + value,5)
    if gene < limits['lr_limits'][0]:
        gene = limits['lr_limits'][0]
    elif gene > limits['lr_limits'][1]:
        gene = limits['lr_limits'][1]

    return gene


def betas_mutation(gene, i, limits={}):
    limits = limits_parser(limits)
    if i == 2:
        value1 = round(npr.uniform(0.05, 0.15), 2)
        value2 = -round(npr.uniform(0.05, 0.15), 2)
        value = choice([value2, value1])
        gene = round(gene + value,2)
        if gene < limits['beta1_limits'][0]:
            gene = limits['beta1_limits'][0]
        if gene > limits['beta1_limits'][1]:
            gene = limits['beta1_limits'][1]

        return gene

    elif i == 3:
        value1 = round(npr.uniform(0.05, 0.15), 3)
        value2 = -round(npr.uniform(0.05, 0.15), 3)
        value = choice([value2, value1])
        gene = round(gene + value,3)
        if gene < limits['beta2_limits'][0]:
            gene = limits['beta2_limits'][0]
        if gene > limits['beta2_limits'][1]:
            gene = limits['beta2_limits'][1]

        return gene


def oneint_mutation(gene, i, limits={}):
    limits = limits_parser(limits)
    if i == 4:
        value = choice([-1,1])
        gene = gene + value
        if gene < limits['n_critic_limits'][0]:
            gene = limits['n_critic_limits'][0]
        elif gene > limits['n_critic_limits'][1]:
            gene = limits['n_critic_limits'][1]

        return gene

    elif i == 5:
        value = choice([-1,1])
        gene = gene + value
        if gene < limits['weight_gp_limits'][0]:
            gene = limits['weight_gp_limits'][0]
        elif gene > limits['weight_gp_limits'][1]:
            gene = limits['weight_gp_limits'][1]

        return gene


# Replacement
def elitism_replacement(current_population, new_population):
    """
    :param current_population: The current population
    :param new_population: The new population
    :return: The new population with the fittest ones
    """

    if current_population.fittest.fitness < new_population.fittest.fitness:
        new_population.solutions[0] = current_population.solutions[-1]
    elif current_population.fittest.fitness < new_population.solutions[-2].fitness:
        new_population.solutions[1] = current_population.solutions[-1]
    if current_population.solutions[-2].fitness < new_population.solutions[-2].fitness:
        new_population.solutions[2] = current_population.solutions[-2]

    return deepcopy(new_population)

