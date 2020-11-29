from copy import deepcopy
from numpy import random
from ydata_synthetic.synthesizers import WGAN_GP
from ydata_synthetic.ga_operators import limits_parser

class Solution:
    """
    Solutions of the population.
    """

    # Constructor
    # ----------------------------------------------------------------------------------------------
    def __init__(self,limits={}):
        self._representation = None
        self._fitness = None
        self._is_fitness_calculated = False
        self._limits = limits_parser(limits)

        #self._representation = self.representation


    # representation
    # ----------------------------------------------------------------------------------------------
    @property
    def representation(self):
        if self._representation is None:
            self.set_representation()
        return self._representation


    def set_representation(self):
        """
        It generates the individual
        """
        limits = self._limits
        solution_representation = []
        for i in range(0, 6):
            if i in [0,4,5]:
                max_val = list(limits.values())[i][1]
                min_val = list(limits.values())[i][0]
                gene = int(random.randint(min_val,max_val))
                solution_representation.append(gene)
            elif i in [1,2,3]:
                max_val = list(limits.values())[i][1]
                min_val = list(limits.values())[i][0]
                gene = random.uniform(min_val, max_val)
                solution_representation.append(gene)

        solution_representation[1] = round(solution_representation[1], 5)
        solution_representation[2] = round(solution_representation[2], 2)
        solution_representation[3] = round(solution_representation[3], 3)

        self._representation = solution_representation


    # Fitness
    # ----------------------------------------------------------------------------------------------

    @property
    def fitness(self):
        return self._fitness

    def calculate_fitness(self, dimension, data, train_args):
        #self._fitness = int(random.randint(0,3000,1))
        parameters = self.representation[:-2]
        parameters = parameters + dimension
        n_critic = self.representation[-2:-1][0]
        weight_gp = self.representation[-1:][0]
        model = WGAN_GP(parameters,n_critic,weight_gp)
        self._fitness = model.train(data, train_args)

    def set_fitness(self,value):
        self._fitness = value


