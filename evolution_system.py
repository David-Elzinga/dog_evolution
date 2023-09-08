import heapq
import random
import bisect
import diptest
import numpy as np
from scipy.stats import truncnorm

class canine:
    def __init__(self, age, sex, tau):
        self.age = age
        self.sex = sex
        self.tau = tau

class system:
    def __init__(self, parms, name):
        self.parms = parms
        self.name = name
        self.census = []

    def simulate(self):

        def trunc_norm(lower_bound, upper_bound, mean, std, size = 1):
            # define the clipping parameters
            a = (lower_bound - mean) / std
            b = (upper_bound - mean) / std
            return(truncnorm.rvs(a, b, loc = mean, scale = std, size = size))

        def initialize(self):
            # create canines (an equal number for each inital tau value), 
            canines = set()
            canines_per_tau = int(self.parms['c'] / len(self.parms['mu_tau']))
            for mu_tau in self.parms['mu_tau']:
                # draw the ages for canines according to fig 2 in doi.org/10.1139/Z07-001, equal prob. of each 
                # sex, and use a truncated normal to get the tau values.
                ages = np.random.choice(a = range(10),
                                        size = canines_per_tau,
                                        p = [0.34, 0.21, 0.15, 0.11, 0.05, 0.05, 0.04, 0.02, 0.02, 0.01])
                sexes = np.random.choice(a = ['male', 'female'],
                                        size = canines_per_tau)
                
                taus = trunc_norm(lower_bound = 0,
                                  upper_bound = 1, 
                                  mean = mu_tau,
                                  std = self.parms['sigma_tau'],
                                  size = canines_per_tau)
                
                canines.update({canine(ages[n], sexes[n], taus[n]) for n in range(canines_per_tau)})
            return(canines)
        
        def hunt(canines, human_food, wild_food):
            
            def feed(hunters, food, target):
                # Create a list of weights based on the target
                hunters = list(hunters)
                weights = [c.tau if target == 'human' else 1 - c.tau for c in hunters]
                
                fed_canines = set()
                while hunters and food > 0:
                    # Check if there are positive weights left
                    if any(weight > 0 for weight in weights):
                        chosen_hunter_index = random.choices(range(len(hunters)), weights=weights, k=1)[0]
                    else:
                        chosen_hunter_index = random.randint(0, len(hunters) - 1)
                    
                    chosen_hunter = hunters.pop(chosen_hunter_index)
                    chosen_weight = weights.pop(chosen_hunter_index)
                    
                    if np.random.rand() < chosen_weight:
                        fed_canines.add(chosen_hunter)
                        food -= 1
                        
                return fed_canines

            # separate canines into those who prefer human food (tau > 0.5) and wild food (t <= 0.5).
            pref_human = {c for c in canines if c.tau > 0.5}
            pref_wild = canines - pref_human

            # perform feeding rounds with different food preferences and categories
            fed_canines = set()

            if len(pref_human) > 0: fed_canines |= feed(pref_human, human_food, 'human') 
            pref_human = pref_human - fed_canines # take those successful out so they don't attempt a second feeding.
            
            if len(pref_wild) > 0: fed_canines |= feed(pref_wild, wild_food, 'wild')   
            pref_wild = pref_wild - fed_canines # take those successful out so they don't attempt a second feeding.
            
            if len(pref_human) > 0: fed_canines |= feed(pref_human, wild_food, 'wild') 
            if len(pref_wild) > 0: fed_canines |= feed(pref_wild, human_food, 'human')
            return fed_canines
        
        def reproduce(canines, total_food):
            def weighted_random_indv(cumsum, rdm_unif):
                return bisect.bisect_left(cumsum, rdm_unif)

            # determine the prob. a female reproduces
            num_males = sum(c.sex == 'male' for c in canines)
            num_females = len(canines) - num_males

            if num_females > 0:
                c_female = self.parms['c'] * num_females / (num_females + num_males)
                L = max(min(self.parms['r']*(1 - num_females/c_female)/(self.parms['q']*self.parms['mu_L']), 1), 0)
            else: L = 0

            # identify the reproductive males
            reproductive_males = [c for c in canines if c.sex == 'male' and c.age >= 2]
            reproducing_males = []

            # identify the reproductive females and determine the number of litters (at most the number
            # of reproductive males).
            reproductive_females = np.array([c for c in canines if c.sex == 'female' and c.age >= 2])
            num_litters = min(np.random.binomial(n = len(reproductive_females), p = L), len(reproductive_males))
            reproducing_females = list(np.random.choice(reproductive_females,
                                                   size = num_litters,
                                                   replace = False))
            
            # pick mating pairs depending on mate preference - shuffle females and males to prevent 
            # ordering from hunting scheme as an extra precaution
            random.shuffle(reproducing_females); random.shuffle(reproductive_males)
            if self.parms['mate_pref'] and num_litters > 0:
                # create a matrix to hold 1 - the (absolute) difference between reproducing females and males
                # small values indicate low relative prob of a mating pair, high values indicate high
                # relative prob of a maiting pair
                reproducing_female_taus = np.array([c.tau for c in reproducing_females])
                reproductive_male_taus = np.array([c.tau for c in reproductive_males])
                similarity_matrix = 1 - np.abs(reproducing_female_taus[:, np.newaxis] - reproductive_male_taus)

                # starting with the first row of the matrix (first female), pick a corresponding male according
                # to the row's normalized probabilities - then remove the male from the reproductive male list and delete 
                # the column from the pair_prob_matrix corresponding to that male and re-normalize
                for dam_idx in range(similarity_matrix.shape[0]):
                    pair_prob = similarity_matrix[dam_idx, :]
                    if sum(pair_prob) > 0: # check to see there's at least one attractive male
                        sire_idx =  np.random.choice(range(len(pair_prob)), p = pair_prob / sum(pair_prob))
                    else:
                        sire_idx = np.random.choice(range(len(pair_prob)))
                    reproducing_males.append(reproductive_males[sire_idx])
                    reproductive_males = np.delete(reproductive_males, sire_idx)
                    if similarity_matrix.shape[1] == 1: break # if that was the last male, stop. 
                    else:
                        similarity_matrix = np.delete(similarity_matrix, sire_idx, axis = 1)
            elif num_litters > 0: # randomly pick the males
                reproducing_males = list(np.random.choice(reproductive_males,
                                                    size=min(num_litters, len(reproductive_males)),
                                                    replace=False))
            
            # determine the litter sizes by generating the total number of pups born, then only 
            # allowing a portion to survive
            litter_sizes = np.round(trunc_norm(lower_bound = 4,
                                              upper_bound = 10,
                                              mean = self.parms['mu_L'],
                                              std = self.parms['sigma_L'], 
                                              size = num_litters)).astype(int)
            litter_sizes = np.random.binomial(n = litter_sizes, p = self.parms['q'])

            # determine the sexes of pups in each litter
            pup_sexes = np.random.choice(['male', 'female'], sum(litter_sizes)).tolist()
            cum_sum_pups = np.cumsum(litter_sizes)
            pup_sexes = np.split(pup_sexes, cum_sum_pups[:-1])
            pup_sexes = [list(litter_sexes) for litter_sexes in pup_sexes]

            # determine the number of mutants in each litter
            num_mutants = np.random.binomial(n = litter_sizes, p = 10**self.parms['expn_m'])

            # iterate through each breeding pair, generate their pups
            for dam, sire, litter_size, sexes, n_mutants in zip(reproducing_females, reproducing_males, litter_sizes, pup_sexes, num_mutants):

                # create mutant pups' tau and non mutatn pups' taud
                pup_taus = np.random.random(size = n_mutants).tolist()
                min_tau = min([dam.tau, sire.tau])
                max_tau = max([dam.tau, sire.tau])
                non_mutant_taus = (np.random.uniform(min_tau, max_tau, size = litter_size - n_mutants) + 
                                   np.random.normal(0, 10**self.parms['expn_sigma_e'], size = litter_size - n_mutants)).tolist()
                pup_taus += np.minimum(np.maximum(non_mutant_taus, 0), 1).tolist()

                # add the pups to the pack
                new_pups = {canine(0, sexes[k], pup_taus[k]) for k in range(litter_size)}
                canines |= new_pups

            return canines



        # generate initial canines - iterate from 30,000ybp to 15,000ybp one year at time, set the 
        # food equal to the carrying capacity, make census to track results
        canines = initialize(self); food = [self.parms['c']]; census = []
        for year_bp in range(30000, 15000, -1):

            # update the amount of food using a truncated normal centered at last years' food and allocate
            # human and wild food sources.
            food.append(int(np.round(trunc_norm(lower_bound = 0,
                                                upper_bound = self.parms['c'],
                                                mean = food[-1],
                                                std = 10**self.parms['expn_sigma_f']))))
            if self.parms['food_scheme'] == 'constant':
                human_food = int(np.round(food[-1]*self.parms['h_max'] / 2))
            else:
                human_food = int(food[-1]*(self.parms['h_max']*(15000 - year_bp)/15000 + self.parms['h_max']))
            wild_food = food[-1] - human_food

            # simulate natural death - for each canine draw a random number on the unit interval, only 
            # keep those whose number is larger than their probability of natural death. 
            u = np.random.uniform(0, 1, len(canines))
            canines = {c for n, c in enumerate(canines) if u[n] > self.parms['nat_death_prob'][c.sex][c.age]}

            # simulate hunting and reproduction
            canines = hunt(canines, human_food, wild_food)
            canines = reproduce(canines, food[-1])

            # if there are less than (or equal to) 3 canines, stop due to invalid dip test results
            if len(canines) <= 3:
                break
            elif year_bp % 10 == 0:
                taus = [c.tau for c in canines]
                _, p_val = diptest.diptest(np.array(taus))
                if self.parms['low_memory_mode']:
                    census.append({
                    'YBP': year_bp,
                    'human_food': human_food,
                    'wild_food': wild_food,
                    'dip_p': p_val})
                else:
                    census.append({
                    'YBP': year_bp,
                    'taus': taus,
                    'ages': [c.age for c in canines],
                    'human_food': human_food,
                    'wild_food': wild_food,
                    'dip_p': p_val})

            # increment the age of all canines
            for c in canines: c.age += 1
        return census