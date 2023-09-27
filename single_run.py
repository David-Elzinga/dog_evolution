import time
import pickle
import numpy as np
import evolution_system

parms = {
    'mu_tau': [0.1],
    'sigma_tau': 0.05,
    'c': 500,
    'expn_sigma_f': 0.3,
    'd': 0.68,
    'food_scheme': 'constant',
    'mate_pref': False,
    'h_max': 1,
    'r': 4.08,
    'mu_L': 6.8,
    'sigma_L': 2.2,
    'expn_m': -2,
    'q': 0.5,
    'expn_sigma_e': -2,
    'mate_pickiness': 0.2, # values close to zero imply little pickiness, values close to 1 imply strong pickiness,
    'low_memory_mode': False
}

parms['nat_death_prob'] = {
    'female': parms['d']*np.array([0, 0.17, 0.16, 0.5, 0.25, 0.5, 0.17, 0.40, 0.40, 1/parms['d']]),
    'male': parms['d']*np.array([0, 0.18, 0.36, 0.19, 0.45, 0.5, 0.5, 0.33, 0.33, 1/parms['d']])
}

t0 = time.time()
test_system = evolution_system.system(parms = parms,
                                      name = "test system")
census = test_system.simulate()
t1 = time.time()
print(t1 - t0)

with open("data/single_run.pickle", "wb") as f:
    pickle.dump({'system': test_system, 'census': census}, f)