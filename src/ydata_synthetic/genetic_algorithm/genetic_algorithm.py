from ydata_synthetic.genetic_algorithm.population import Population
from ydata_synthetic.genetic_algorithm.solution import Solution
from ydata_synthetic.genetic_algorithm.ga_operators import singlepoint_crossover, adapted_mutation, elitism_replacement, roulette_wheel

from time import time
from copy import deepcopy
from random import random

default_params = {
    "population_size": 20,
    "cross_probability": 0.5,
    "mutation_probability": 0.1,
    "selection_approach": roulette_wheel,
    "crossover_approach": singlepoint_crossover,
    "mutation-approach": adapted_mutation,
    "replacement-approach": elitism_replacement
}

class GeneticAlgorithm:
    """
    1. Initial population
    2. Fitness function
    3. Selection
    4. Crossover
    5. Mutation
    6. Replacement
    """

    # Constructor
    # ---------------------------------------------------------------------------------------------
    def __init__(self, dimension, data, train_args, params=default_params,metric='kl', limits={}, population=None, saveit=True):
        self._parse_params(params)
        self._limits = limits
        self.dimension = dimension
        self.data = data
        self.train_args = train_args
        self.metric = metric
        self._saveit = saveit

        self._population = population
        self._fittest = None

        self._top_5 = []

    # search
    # ---------------------------------------------------------------------------------------------
    def search(self):
        """
            Genetic Algorithm Search Algorithm
            1. Initial population

            2. Repeat n generations )(#1 loop )

                2.1. Repeat until generate the next generation (#2 loop )
                    1. Selection
                    2. Try Apply Crossover (depends on the crossover probability)
                    3. Try Apply Mutation (depends on the mutation probability)

                2.2. Replacement

            3. Return the best solution
        """
        select = self._selection_approach
        cross = self._crossover_approach
        mutate = self._mutation_approach
        replace = self._replacement_approach


        self._generation = 0

        # 1. Initial population
        start_time = time()
        if self._population==None:
            print('Creating the first population.')
            self._population = Population(self._population_size, self.inizialize_solutions(self._population_size))
            self._update_top_5(0, self._population, time() - start_time)
            print(f'First population created. Time: {time()-start_time}')
            self._population.save('./pops/initial_pop.pkl')

        ## 2. Repeat n generations )(#1 loop ) ##==============
        for generation in range(0, self._number_of_generations):

            new_population = Population(maximum_size=self._population_size, solution_list=[])
            start_time = time()
            print(f'\n \n Generation # {generation+1} \n ')

            # 2.1. Repeat until generate the next generation (#2 loop )
            while new_population.has_space:

                # 2.1.1. Selection
                parent1, parent2 = select(solutions=self._population.solutions)
                offspring1 = deepcopy(parent1)
                offspring2 = deepcopy(parent2)

                # 2.1.2. Apply crossover with probability
                if self.apply_crossover:
                    offspring1, offspring2 = cross(parent1, parent2)

                # 2.1.3. Apply Mutation with probability
                if self.apply_mutation:
                    print(offspring1.representation)
                    offspring1 = mutate(offspring1)

                if self.apply_mutation:
                    print(offspring2.representation)
                    offspring2 = mutate(offspring2)

                # Add the offsprings in the new population (New Generation)
                print(f'Calculating the fitness of a new individual. Generation: {generation+1}')
                print(f'offspring parameters: {offspring1.representation}')
                offspring1.calculate_fitness(self.dimension,self.data,self.train_args)
                new_population.solutions.append(offspring1)

                if new_population.has_space:
                    print(f'Calculating the fitness of a new individual. Generation: {generation+1}')
                    print(f'offspring parameters: {offspring2.representation}')
                    offspring2.calculate_fitness(self.dimension,self.data,self.train_args)
                    new_population.solutions.append(offspring2)

            new_population.sort()

            # 2.2. Replacement
            self._population.sort()
            self._population = replace(self._population, new_population)
            fittest = self._population.fittest
            if self._saveit ==True:
                self._population.save(f'pops/pop_number_{generation+1}.pkl')
            print(f'The best ones have been replaced. Fitness of the fittest: {fittest.fitness}')
            self._update_top_5(generation, self._population, time() - start_time)

        return fittest, self._top_5

    @property
    def apply_crossover(self):
        chance = random()
        return chance < self._crossover_probability

    @property
    def apply_mutation(self):
        chance = random()
        return chance < self._mutation_probability

    # initialize
    # ---------------------------------------------------------------------------------------------
    def _parse_params(self, params):
        self._params = params
        # set Genetic Algorithm Behavior
        # Initialization signature: <method_name>( problem, population_size ):

        self._population_size = 5
        if "population_size" in params:
            self._population_size = params["population_size"]

        self._crossover_probability = 0.5
        if "cross_probability" in params:
            self._crossover_probability = params["cross_probability"]

        self._mutation_probability = 0.8
        if "mutation_probability" in params:
            self._mutation_probability = params["mutation_probability"]

        self._number_of_generations = 5
        if "number_generations" in params:
            self._number_of_generations = params["number_generations"]

        self._selection_approach = roulette_wheel
        if "selection_approach" in params:
            self._selection_approach = params["selection_approach"]

        self._crossover_approach = singlepoint_crossover
        if "cross_approach" in params:
            self._crossover_approach = params["cross_approach"]

        self._mutation_approach = adapted_mutation
        if "mutation_approach" in params:
            self._mutation_approach = params["mutation_approach"]

        self._replacement_approach = elitism_replacement
        if "replacement_approach" in params:
            self._replacement_approach = params["replacement_approach"]

    def _update_top_5(self, iteration, population, time):

        self._top_5 = []
        top5 = sorted(population.solutions, key=lambda c: c.fitness)[:6]

        for individual in top5:
            self._top_5.append((iteration, individual.representation, individual.fitness, time))

        return self._top_5

    def inizialize_solutions(self, population_size):
        """
        :arg: population_size: Size of the population
        :return: The list of the solution of the population
        """
        if population_size is None:
            population_size=self._population_size

        solution_list=[]
        for i in range(population_size):
            solution = Solution(self._limits,self.metric)
            solution.calculate_fitness(self.dimension,self.data,self.train_args)
            solution_list.append(solution)

        return solution_list