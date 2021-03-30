from pickle import dump, load

class Population:
    """
    Population
    """

    # ---------------------------------------------------------------------------------------------
    def __init__(self, maximum_size, solution_list):
        self._max_size = maximum_size
        self._list = solution_list
        self._fittest = None
        self._sorted = False

    # ---------------------------------------------------------------------------------------------
    @property
    def fittest(self):
        self.sort()
        if len(self._list) > 0:
            return self._list[-1]
        return None

    @property
    def least_fit(self):
        self.sort()
        if len(self._list) > 0:
            return self._list[0]
        return None

    def get_representations(self):
        represent = []
        for repr in self._list:
            represent.append(repr.fitness)
        return represent

    def get_fitnesses(self):
        fit_list = []
        for fit in self._list:
            fit_list.append(fit.fitness)
        return fit_list

    @property
    def size(self):
        return len(self._list)

    @property
    def has_space(self):
        return len(self._list) < self._max_size

    @property
    def is_full(self):
        return len(self._list) >= self._max_size

    def add(self, solution):
        self._list.append(solution)

    def get(self, index):
        """
        It returns a solution of the population according to the index
        """
        if index >= 0 and index < len(self._list):
            return self._list[index]
        else:
            return None

    @property
    def solutions(self):
        """
        Solution list (of the population)
        """
        return self._list

    #
    def sort(self):
        """
        it sorts the population in ascending order of fittest solution (lowest fitness)
        """

        for i in range(0, len(self._list)):
            for j in range(i, len(self._list)):
                if self._list[i].fitness < self._list[j].fitness:
                    swap = self._list[j]
                    self._list[j] = self._list[i]
                    self._list[i] = swap

        self._sorted = True

    def save(self, path: str):
        with open(path, "wb") as f:
            dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return load(f)
