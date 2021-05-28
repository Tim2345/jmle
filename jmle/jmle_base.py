import numpy as np
from time import time

class JMLE_base:

    def __init__(
            self, resp_mat, person_status, item_status, global_person_status, global_item_status,
            anchor_persons, anchor_items, logit_conv, resid_conv, n_iter
    ):
        '''

        :param resp_mat: np.array, response matrix of 1, 0. axis 0=persons, axis1=items
        :param person_status: np.array 1d, 0=anchor persons, 1=uncalibrated persons
        :param item_status: np.array 1d, 0=anchor items, 1=uncalibrated item
        :param global_person_status: float, 0 if no persons are anchored and 1 if persons are anchored
        :param global_item_status: float, 0 if no items are anchored and 1 if items are anchored
        :param anchor_persons: np.array, anchor persons values at corresponding index. Unanchored perons must = 0
        :param anchor_items: np.array, anchor item values at corresponding index. Unanchored items must = 0
        :param logit_conv: float, largest change in updates across people and items. once smaller than this value, the algorithm stops
        :param resid_conv: float, largest residual across people and items. once smaller than this value, the algorithm stops
        :param n_iter: int, total number of iterations before algorithm stops
        '''
        if not resp_mat.shape[0] == len(person_status) == len(anchor_persons):
            raise ValueError("resp_mat along axis 0, __len__ of person_status and __len__ of anchor_persons must match.")

        if not resp_mat.shape[1] == len(item_status) == len(anchor_items):
            raise ValueError("resp_mat along axis 1, __len__ of item_status and __len__ of anchor_items must match.")



        self.resp_mat = resp_mat
        self.person_status = person_status
        self.item_status = item_status
        self.global_person_status = global_person_status
        self.global_item_status = global_item_status
        self.anchor_persons = anchor_persons
        self.anchor_items = anchor_items
        self.logit_conv = logit_conv
        self.resid_conv = resid_conv
        self.n_iter = n_iter
        self.variances_matrix = np.vectorize(self.variances_matrix)

    ########################### MATRIX FUNCTIONS ##############################
    ######### define Rasch function to create probabilities matrix
    def rasch(self, person, item):
        '''
        method to give rasch probability of person answering item correctly
        :param person: logit ability of the person
        :param item: logit difficulty of the item
        :return: rasch probability of person answering item correctly
        '''
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

        #self.variances_matrix = np.vectorize(self.variances_matrix)

    ##### define funciton to create residuals matrix
    def residuals_matrix(self, response_matrix, probabilities_matrix):
        '''
        input: np.matricies of responses and probabilities
        function: calculates the residuals of the expected values and responses
        output: np.matrix of residuals
        '''
        return response_matrix - probabilities_matrix

        ######################### ESTIMATOR FUNCTIONS #############################

    ## Calculate initial estimates
    def initial_estimates(self, response_matrix, anchor_difficulties, axis):
        prop = response_matrix.mean(axis=axis)
        init_diffs = np.log((1 - prop) / prop)

        init_diffs_mean = np.mean(init_diffs)
        adjusted_difficulties = init_diffs - init_diffs_mean
        np.put(adjusted_difficulties,
               np.where(anchor_difficulties != 0),
               anchor_difficulties[np.where(anchor_difficulties != 0)])

        return adjusted_difficulties


    ## Update estimates
    def update_estimates(self, difficulties, residual_matrix, variance_matrix,
                         global_status, item_status, axis):
        '''
        input: np.array of current item difficulties
               matrix of residuals
               matrix of variances
        function: calculates the updated task difficulties
        output: np.array of updated item difficulties
        '''
        sum_of_residuals = -1 * residual_matrix.sum(axis=axis)
        sum_of_variance = -1 * variance_matrix.sum(axis=axis)

        new_difficulties = difficulties - item_status * sum_of_residuals / sum_of_variance
        new_difficulties_mean = global_status * np.mean(new_difficulties)
        adjusted_difficulties = new_difficulties - new_difficulties_mean

        return adjusted_difficulties


    ##### define function to check maximum logit change
    def max_logit_change(self, old_abilities, old_difficulties, new_abilities, new_difficulties):
        old = np.concatenate((old_abilities, old_difficulties))
        new = np.concatenate((new_abilities, new_difficulties))
        return (max(abs(old - new)))


    def estimate(self):
        start = time()

        #TODO must find out why there is a +1 here (Scaling from inital version?)
        resid_mat = self.resid_conv + 1
        logit_change = self.logit_conv + 1

        ######## start estimation
        abilities = self.initial_estimates(self.resp_mat, self.anchor_persons,
                                           self.global_person_status, self.person_status)

        difficulties = self.initial_estimtates(self.resp_mat, self.anchor_items,
                                                 self.global_item_status, self.item_status)
        ## initialize printables
        iter_count = 1

        while logit_change > self.logit_conv and np.max(resid_mat) > self.resid_conv:
            # assign old abilities to check convergence at end of iteration
            abilities_old = abilities
            difficulties_old = difficulties

            # calculate matrices used in iteration
            prob_mat = self.probability_matrix(abilities, difficulties)
            var_mat = self.variances_matrix(prob_mat)
            resid_mat = self.residuals_matrix(self.resp_mat, prob_mat)

            # re-estimate difficulties and abilities
            abilities = self.update_estimates(abilities, resid_mat, var_mat,
                                              self.global_person_status, self.person_status)

            difficulties = self.update_estimates(difficulties, resid_mat, var_mat,
                                                    self.global_item_status, self.item_status)

            logit_change = self.max_logit_change(abilities_old, difficulties_old,
                                                 abilities, difficulties)

            iter_count += 1

        finish = time()

        self.abilities_logits = abilities
        self.difficulties_logits = difficulties
        self.n_iterations = iter_count
        self.n_persons_n_items = self.response_matrix.shape
        self.estimation_time = finish - start
        self.estimator_runs = self.estimator_runs + 1
        #self.person_sem = self.sem(var_mat, 'persons')
        #self.item_sem = self.sem(var_mat, 'items')
        #self.person_infit = self.infit(resid_mat, var_mat, 'persons')
        #self.item_infit = self.infit(resid_mat, var_mat, 'items')
        #self.person_outfit, self.item_outfit = self.outfit(resid_mat, var_mat)
        #self.item_disp_logits = self.displacement(resid_mat, var_mat, 'items')

        print('Persons - items:', *self.n_persons_n_items)
        print('Iterations:', self.n_iterations)
        print('Estimation time: {} seconds'.format(np.round(self.estimation_time, 2)))

        ## post estimation reporting
        #self.report_tables.append(self.reporting_table())

