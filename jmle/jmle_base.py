import numpy as np

class JMLE:

    def __init__(self, logit_conv, resid_conv, n_iter):
        self.logit_conv = logit_conv
        self.resid_conv = resid_conv
        self.n_iter = n_iter

    ########################### MATRIX FUNCTIONS ##############################
    ######### define Rasch function to create probabilities matrix
    def rasch(self, person, item):
        it_per = person - item
        return np.exp(it_per) / (1 + np.exp(it_per))

    ## create expected values matrix
    def probability_matrix(self, abilities, difficulties):
        '''
        input: np.arrays of abilties and difficulties
        function: calculates the Rasch probabilites of
                  each person responding to each item
        output: np.matrix of person item probabilities
        '''
        n_items = len(difficulties)
        n_persons = len(abilities)

        # this was originally masked
        probabilities = np.array([self.rasch(person, item) for person in abilities for item in difficulties])

        return probabilities.reshape(n_persons, n_items)

    ####### define function to create variances matrix
    def variances_matrix(self, probabilities_matrix):
        '''
        function must be vectorized using np.vectorize()
        input: np.arrays of abilties and difficulties
        function: calculates the Rasch probabilites of
                  each person responding to each item
        output: np.matrix of person item probabilities
        '''
        return probabilities_matrix * (1 - probabilities_matrix)



    ##### define funciton to create residuals matrix
    def residuals_matrix(self, response_matrix, probabilities_matrix):
        '''
        input: np.matricies of responses and probabilities
        function: calculates the residuals of the expected values and responses
        output: np.matrix of residuals
        '''
        return response_matrix - probabilities_matrix

        ######################### ESTIMATOR FUNCTIONS #############################

    ## Calculate initial difficulties
    def initial_difficulties(self, response_matrix, anchor_difficulties,
                             global_item_status, item_status):
        prop = response_matrix.mean(axis=0)
        init_diffs = np.log((1 - prop) / prop)

        init_diffs_mean = np.mean(init_diffs)
        adjusted_difficulties = init_diffs - init_diffs_mean
        np.put(adjusted_difficulties,
               np.where(anchor_difficulties != 0),
               anchor_difficulties[np.where(anchor_difficulties != 0)])

        return adjusted_difficulties

    ## Calculate initial abilities
    def initial_abilities(self, response_matrix):
        prop = response_matrix.mean(axis=1)
        return np.log(prop / (1 - prop))

    ## Update item difficulties
    def update_estimates(self, difficulties, residual_matrix, variance_matrix,
                         global_status, item_status):
        '''
        input: np.array of current item difficulties
               matrix of residuals
               matrix of variances
        function: calculates the updated task difficulties
        output: np.array of updated item difficulties
        '''
        sum_of_residuals = -1 * residual_matrix.sum(axis=0)
        sum_of_variance = -1 * variance_matrix.sum(axis=0)

        new_difficulties = difficulties - item_status * sum_of_residuals / sum_of_variance
        new_difficulties_mean = global_status * np.mean(new_difficulties)
        adjusted_difficulties = new_difficulties - new_difficulties_mean

        return adjusted_difficulties

    ## update person abilities
    def update_abilities(self, abilities, residual_matrix, variance_matrix):
        '''
        input: np.array of current person abilities
               matrix of residuals
               matrix of variances
        function: calculates the updated person abilities
        output: np.array of updated person abilities
        '''
        sum_of_residuals = residual_matrix.sum(axis=1)
        sum_of_variance = -1 * variance_matrix.sum(axis=1)
        # abilities do not need to be adjusted
        return (abilities - sum_of_residuals / sum_of_variance)

    ##### define function to check maximum logit change
    def max_logit_change(self, old_abilities, old_difficulties, new_abilities, new_difficulties):
        old = np.concatenate((old_abilities, old_difficulties))
        new = np.concatenate((new_abilities, new_difficulties))
        return (max(abs(old - new)))

self.variances_matrix = np.vectorize(self.variances_matrix)