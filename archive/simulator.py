import numpy as np
import itertools
from scipy.stats import truncnorm
import time
import bisect
import diptest


# Still not using some of my subfunctions? Getting negative tau values? Problem with cum sum not finding first element

class system:
    def __init__(self, p, name):
        '''
        Initialize a system with a parameter set and a name 
        '''
        self.p = p
        self.name = name

class canine:
    def __init__(self, age, sex, tau):
        '''
        Initialize a canine given the age, sex, tau (human tolerance). 
        '''
        self.age = age
        self.sex = sex
        self.tau = tau

def simulate(s):
    '''
    Run the evolution simulation
    '''

    def hunt(canines, human_food, wild_food):

        def perform_feeding_round(canines, food, sort_tau_incr, food_target):

            sorted_canines = sorted(canines, key=lambda x: x.tau, reverse=sort_tau_incr)
            cum_sum = np.cumsum([x.tau if food_target == 'human' else 1 - x.tau for x in sorted_canines])
            fed_canines = set()

            while sorted_canines and food > 0:
                
                if cum_sum[-1] <= 0:
                    idx = 0
                else:
                    u = np.random.rand() * cum_sum[-1]
                    idx = bisect.bisect_left(cum_sum, u) #idx = np.searchsorted(cum_sum, u)
                selected_canine = sorted_canines[idx]

                if (np.random.rand() < selected_canine.tau) == sort_tau_incr:
                    fed_canines.add(selected_canine)
                    food -= 1

                sorted_canines.pop(idx)
                cum_sum[idx+1:] -= (selected_canine.tau if food_target == 'human' else 1 - selected_canine.tau)
                mask = np.arange(len(cum_sum)) != idx; cum_sum = cum_sum[mask]
            return fed_canines
    
        # Separate the canines into those who prefer human food (tau > 0.5) and wild food (t <= 0.5).
        pref_human = {c for c in canines if c.tau > 0.5}
        pref_wild = canines - pref_human

        # Perform feeding rounds with different food preferences and categories
        fed_canines = set()
       
        fed_canines |= perform_feeding_round(pref_human, human_food, True, 'human') 
        pref_human = pref_human - fed_canines # take those successful out so they don't attempt a second feeding.
        
        fed_canines |= perform_feeding_round(pref_wild, wild_food, False, 'wild')   
        pref_wild = pref_wild - fed_canines # take those successful out so they don't attempt a second feeding.

        fed_canines |= perform_feeding_round(pref_human, wild_food, False, 'wild') 

        fed_canines |= perform_feeding_round(pref_wild, human_food, True, 'human')
            
        return fed_canines

    def reproduce(canines, total_food, s):
        
        def weighted_random_indv(w):
            cumsum = np.cumsum(w)
            rdm_unif = np.random.random()
            return bisect.bisect_left(cumsum, rdm_unif) #np.searchsorted(cumsum, rdm_unif)
        
        num_canines = len(canines)
        num_males = sum(c.sex == 'm' for c in canines)
        num_females = num_canines - num_males

        rep_males = [c for c in canines if c.sex == 'm' and c.age >= 2]
        rep_females = np.array([c for c in canines if c.sex == 'f' and c.age >= 2])
        male_taus = np.array([c.tau for c in rep_males])

        if num_females > 0:
            c_female = s.p['c'] * num_females / (num_females + num_males)
            L =  s.p['r']*(1 - num_females/c_female)/(s.p['q'] * s.p['mu_L'])
        else:
            L = 0

        picked_females_idx = np.random.random(size=len(rep_females)) < L
        picked_females = rep_females[picked_females_idx]
        rng.shuffle(picked_females)
        num_litters = len(picked_females)

        if not s.p['mate_pref']:
            picked_males = np.random.choice(rep_males, size=min(num_litters, len(rep_males)), replace=False)
        else:
            picked_males = []
            for f in picked_females:
                if len(rep_males) == 0: # if there are no more males left to reproduce, stop.
                    break
                else:
                    tau_diff = np.abs(male_taus - f.tau)
                    if (1 - tau_diff).sum() == 0:
                        picked_male_idx = np.random.choice(range(len(rep_males)))
                    else:
                        picked_male_idx = weighted_random_indv((1 - tau_diff) / ((1 - tau_diff).sum()))

                    picked_male = rep_males[picked_male_idx]
                    rep_males = np.delete(rep_males, picked_male_idx)
                    mask = np.arange(len(male_taus)) != picked_male_idx; male_taus = male_taus[mask] 

                picked_males.append(picked_male)

        # Determine litter sizes
        a, b = (4 - s.p['mu_L']) / s.p['sigma_L'], (10 - s.p['mu_L']) / s.p['sigma_L']
        litter_sizes = np.rint(truncnorm.rvs(a, b, loc=s.p['mu_L'], scale=s.p['sigma_L'], size=num_litters)).astype(np.int32)

        # Determine the number of pups that survive in each litter.
        num_surv_pups = np.random.binomial(litter_sizes, s.p['q'])

        # Determine the sexes of pups in each litter. 
        pup_sexes = np.random.choice(['m', 'f'], sum(num_surv_pups)).tolist()
        cum_sum_pups = np.cumsum(num_surv_pups)
        pup_sexes = np.split(pup_sexes, cum_sum_pups[:-1])
        pup_sexes = [list(litter_sexes) for litter_sexes in pup_sexes]

        # Determine the number of mutants in each litter.
        num_mutants = np.random.binomial(num_surv_pups, 10**s.p['expn_m'])

        for dam, sire, num_surv_pups, pup_sexes, num_mutants in zip(picked_females, picked_males, num_surv_pups, pup_sexes, num_mutants):

            # Create taus for mutant pups. 
            pup_taus = np.random.random(size=num_mutants).tolist()

            # Create taus for non mutant pups. 
            min_tau = min([dam.tau, sire.tau])
            max_tau = max([dam.tau, sire.tau])
            non_mutant_taus = np.random.uniform(min_tau, max_tau, size=num_surv_pups - num_mutants) + np.random.normal(0, 10**s.p['expn_sigma_e'], size = num_surv_pups - num_mutants)
            pup_taus += np.minimum(np.maximum(non_mutant_taus, 0), 1).tolist()

            # Add pups to the pack.
            new_pups = {canine(0, pup_sexes[k], pup_taus[k]) for k in range(num_surv_pups)}
            canines |= new_pups
       
        return canines

    ### Seed
    rng = np.random.default_rng(seed = s.p['seed'])

    ### Initialize

    # Create a list to track the total amount of food. Initialize it with the population of wolves.
    food = [s.p['c']]
    
    # Create an empty set to hold the canines
    canines = set()

    # We draw our initial canine population from an abritrary number of distributions equally. Each of these
    # distributions is a truncated normal. Normally we only have one distribution, but for verification we want this flexibility. 
    num_wolves_per_dist = int(s.p['c'] / len(s.p['mu_tau']))

    for dist_mean in s.p['mu_tau']:
        
        # Draw the ages for these wolves (using Fig 2 in doi.org/10.1139/Z07-001 where we impute age
        # 8 to assure the sum is 1.)
        ages = rng.choice(range(10), s.p['c'], p = [0.34, 0.21, 0.15, 0.11, 0.05, 0.05, 0.04, 0.02, 0.02, 0.01])

        # Draw the sexes for these wolves
        sexes = rng.choice(['m', 'f'], num_wolves_per_dist, p = [0.5, 0.5])

        # Calculate the "clip" parameters for the truncated distribution. Draw their human tolerance values
        a, b = (0 - dist_mean) / s.p['sigma_tau'], (1 - dist_mean) / s.p['sigma_tau']
        taus =  truncnorm.rvs(a, b, dist_mean, s.p['sigma_tau'], size = num_wolves_per_dist, random_state = s.p['seed'])

        # Create the canines! 
        canines.update({canine(ages[k], sexes[k], taus[k]) for k in range(num_wolves_per_dist)})

    census = []

    # Iterate from 30,000 YBP to 15,000 YBP
    t0 = time.time()
    for year in range(30000, 15000, -1):

        # Draw a new amount of total food based on last year's amount  
        a, b = -food[-1]/10**s.p['expn_sigma_f'], (s.p['c'] - food[-1])/10**s.p['expn_sigma_f']
        food.append(int(np.round(truncnorm.rvs(a, b, food[-1], 10**s.p['expn_sigma_f'], random_state = year + s.p['seed']))))

        # Allocate the food as either human food, or wild food according to the food scheme 
        if s.p['food_scheme'] == 'constant':
            human_food = int(np.round(food[-1]*s.p['h_max']/2))
        else:
            human_food = int((s.p['h_max']*(15000 - year)/15000 + s.p['h_max'])*food[-1])
        wild_food = food[-1] - human_food

        # Draw a random number on the unit interval for each canine. Simualting natural death,
        # only retain the canines whose random number exceeds their probability of natural death, 
        # which depends on their sex and age.
        u = rng.random(size=len(canines))
        canines = {c for n, c in enumerate(canines) if u[n] > s.p['nat_death_prob'][c.sex][c.age]}

        # Hunt
        canines = hunt(canines, human_food, wild_food)

        # If there are less than 3 wolves, stop (invalid dip test results). 
        if len(canines) <= 3:
            break

        # Reproduce
        canines = reproduce(canines, food[-1], s)

        # Store
        if year % 10 == 0:

            taus = [c.tau for c in canines]
            _, p_val = diptest.diptest(np.array(taus))

            census.append({
                'YBP':year,
                'taus':taus,
                'ages':[c.age for c in canines],
                'human_food':human_food,
                'wild_food':wild_food,
                'dip_p': p_val
            })

        # Increment the age of all canines
        for c in canines: c.age += 1

    t1 = time.time()
    print(t1-t0)
    return census


