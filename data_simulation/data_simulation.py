import numpy as np

class DataSimulator:
    def __init__(self, n_items, n_persons):

        self.n_items = n_items
        self.n_persons = n_persons

    def sample_items_and_persons(self, dist, *args, **kwargs):
        '''
        simple wrapper method for all numpy  random distribution methods
        https://numpy.org/doc/1.16/reference/routines.random.html
        used to generate diffictulies and abilities for simulated items and persons

        '''

        # prepare args - convert to a string for pass to eval()
        arguments = [str(val) for val in [*args]]
        arguments = ','.join(arguments)

        # prepare kwargs - convert to a string for pass to eval()
        kwarguments = []
        for key, arg in kwargs.items():
            kwarguments.append(key + '=' + str(arg))
        kwarguments = ','.join(kwarguments)

        # set comma to divide args and kwargs in eval() if both present
        if args and kwargs:
            div = ','
        else:
            div = ''

        # produce distribution
        distribution = eval('np.random.{}({}{}{})'.format(dist, arguments, div, kwarguments))

        return distribution

    def rasch(self, abilities, difficulties):
        '''
        Method to generate the rasch probabilities given abilities and difficulties
        :param abilities: np.array, or scalar of person abilities
        :param difficulties: np.array or scalar of item difficulties
        :return: np.array or scalar of probability of person(s) of given ability responding correctly to item(s) of given difficulty
                 according to the Rasch model
        '''
        log_diff = np.array(abilities) - np.array(difficulties)
        prob = np.exp(log_diff) / (1 + np.exp(log_diff))

        return prob

    def rasch_response_matrix(self, persons, items, probabilities=False):
        '''
        Method to simulate response matrix given persons and items. This method uses the rasch model to output probabilties
        which are then used to generate probabilistic response patterns using numpy.binomial
        :param persons: np.array, or scalar of person abilities
        :param items: np.array or scalar of item difficulties
        :param probabilities, bool, if probabilities=True, returns tuple of response matrix and probabilties.
        :return: np.array,response matrix of 1, 0 (correct, incorrect) axis=0 - persons, axis=1 - items.
        '''
        probs = np.array([self.rasch(person, item) for person in persons for item in items])
        probs = probs.reshape(len(persons), len(items))

        response_matrix = np.random.binomial(1, probs)

        if not probabilities:
            return response_matrix
        else:
            return (response_matrix, probs)

    def anchor_status(self, size, n_anchors=None, idx=None):
        '''
        Method to simulate anchor status vector used in JMLE. 0 represents an anchor and 1 represents item for calibration.
        if n_anchors is passed, idx must not be and vice versa. if n_anchors is passed, this number of items is randomly selected
        as an anchor. if idx is passed, the indexes are used to selectt he anchor items.
        :param size: number of items in test - len(items)
        :param n_anchors: int, number of anchors
        :param idx: list or np.array of indexes to be used as anchor items
        :return: np.array of 1,0 where 1=item for calibration and 0=anchor_item
        '''
        if n_anchors and idx:
            raise ValueError("'n_anchors' and 'idx' are not valid arguments together. Please choose only one.")

        if n_anchors:
            idx = np.random.choice(np.arange(size), n_anchors, replace=False)

        item_status = np.ones(size)
        item_status[idx] = 0

        return item_status


