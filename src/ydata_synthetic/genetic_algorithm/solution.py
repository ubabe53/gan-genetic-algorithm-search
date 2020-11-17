from copy import deepcopy
from numpy import random
from ydata_synthetic.genetic_algorithm.encoding import encoder
from ydata_synthetic.synthesizers import WGAN_GP

class Solution:
    """
    Solutions of the population.
    """

    # Constructor
    # ----------------------------------------------------------------------------------------------
    def __init__(self):
        self._representation = None
        self._decoded_repr = None
        self._fitness = None
        self._is_fitness_calculated = False

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
        solution_representation = []
        for _ in range(0, 6):
            element = int(random.randint(1,6))
            solution_representation.append(element)
        self._representation = solution_representation


    # encoder
    # ----------------------------------------------------------------------------------------------
    @property
    def decode(self):
        if self._decoded_repr is None:
            self.do_decode()
        return self._decoded_repr


    def do_decode(self):
        """
        decode the representation into the phenotype
        """
        self._decoded_repr = encoder(self.representation)

    # Fitness
    # ----------------------------------------------------------------------------------------------

    @property
    def fitness(self):
        return self._fitness

    def calculate_fitness(self, dimension, data, train_args):  ##TODO Here you gotta implement the WGAN-GP
        #self._fitness = int(random.randint(0,3000,1))
        parameters = self.decode[:-2]
        parameters = parameters + dimension
        n_critic = self.decode[-2:-1][0]
        weight_gp = self.decode[-1:][0]
        model = WGAN_GP(parameters,n_critic,weight_gp)
        self._fitness = model.train(data, train_args)

    def clone(self):
        return deepcopy(self)

