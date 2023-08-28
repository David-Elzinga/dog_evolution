import simulator
import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib
import matplotlib.colors as mcolors
import seaborn as sns

### Simualations


# First system - forced dogs
p = {
    'seed': 0,
    'c': 500,
    'mu_tau': [0.1],
    'sigma_tau': 0.05,
    'expn_sigma_f': 0.3,
    'expn_m': -2,
    'expn_sigma_e': -1,
    'r': 4.08,
    'mu_L': 6.8,
    'sigma_L': 2.2,
    'q': 0.5,
    'food_scheme': 'increasing',  # if constant, half the value
    'mate_pref': False,
    'h_max': 1,
    'd': 0.68,
}

p['nat_death_prob'] = {
    'f': p['d']*np.array([0, 0.17, 0.16, 0.5, 0.25, 0.5, 0.17, 0.40, 0.40, 1/p['d']]),
    'm': p['d']*np.array([0, 0.18, 0.36, 0.19, 0.45, 0.5, 0.5, 0.33, 0.33, 1/p['d']])
}


if os.path.isfile("data/verification/forced_dogs.pickle"):
    with open("data/verification/forced_dogs.pickle", "rb") as f:
        x = pickle.load(f)
    
    verification_system_1 = x['system']
    census_system_1 = x['census']
else:
    verification_system_1 = simulator.system(p, 'forced_dogs')
    census_system_1 = simulator.simulate(verification_system_1)
    with open("data/verification/forced_dogs.pickle", "wb") as f:
        pickle.dump({'system':verification_system_1, 'census':census_system_1}, f)


# Second system: Keeping two populations with mate preference

p = {
    'seed': 0,
    'c': 500,
    'mu_tau': [0.5],
    'sigma_tau': 2,
    'expn_sigma_f': 0.3,
    'expn_m': -2,
    'expn_sigma_e': -1,
    'r': 4.08,
    'mu_L': 6.8,
    'sigma_L': 2.2,
    'q': 0.5,
    'food_scheme': 'constant',  # if constant, half the value
    'mate_pref': True,
    'h_max': 1,
    'd': 0.68,
}

p['nat_death_prob'] = {
    'f': p['d']*np.array([0, 0.17, 0.16, 0.5, 0.25, 0.5, 0.17, 0.40, 0.40, 1/p['d']]),
    'm': p['d']*np.array([0, 0.18, 0.36, 0.19, 0.45, 0.5, 0.5, 0.33, 0.33, 1/p['d']])
}

if os.path.isfile("data/verification/two_pops.pickle"):
    with open("data/verification/two_pops.pickle", "rb") as f:
        x = pickle.load(f)
    
    verification_system_2 = x['system']
    census_system_2 = x['census']
else:
    verification_system_2 = simulator.system(p, 'two_pops')
    census_system_2 = simulator.simulate(verification_system_2)
    with open("data/verification/two_pops.pickle", "wb") as f:
        pickle.dump({'system':verification_system_2, 'census':census_system_2}, f)

### Plotting

# First plot: KDE of tau values at five 100 year periods in the forced dogs situation. 
fig, ax = plt.subplots()
years = [3000, 2500, 2000, 1501]
years_labels = ['30,000 YBP', '25,000 YBP', '20,000 YBP', '15,000 YBP']
line_styles = ['solid', 'dashed', 'dashdot', 'dotted']

for year, lab, ls  in zip(years, years_labels, line_styles):
    tau_values = census_system_1[3000 - year]['taus']
    sns.kdeplot(tau_values, ax = ax, label = lab, cut = 0, clip = (0,1), linestyle = ls, linewidth = 2, color = 'black')

ax.set_xlabel('Human Tolerance, ' + r'$\tau$', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.legend(loc='best', fontsize=10)
fig.savefig('figures/verification/two_pops.pdf')

# Second plot: KDE of tau values at five 100 year periods in the two populations situation. 
fig, ax = plt.subplots()
years = [3000, 2999, 2998, 2997]
years_labels = ['First Gen.', 'Second Gen.', 'Third Gen.', 'Fourth Gen.']
line_styles = ['solid', 'dashed', 'dashdot', 'dotted']

for year, lab, ls  in zip(years, years_labels, line_styles):
    tau_values = census_system_2[3000 - year]['taus']
    sns.kdeplot(tau_values, ax = ax, label = lab, cut = 0, clip = (0,1), linestyle = ls, linewidth = 2, color = 'black')

ax.set_xlabel('Human Tolerance, ' + r'$\tau$', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.legend(loc='best', fontsize=10)
fig.savefig('figures/verification/forced_dogs.pdf')
plt.show()
