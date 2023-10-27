import os
import pickle
import matplotlib
import numpy as np
import seaborn as sns
import evolution_system
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde

## first system, forced dogs - force h_max to go from 0 to 1 to demonstrate that dogs 
## will form. 
parms = {
    'c': 500,
    'mu_tau': [0.1],
    'sigma_tau': 0.05,
    'expn_sigma_f': 0.3,
    'expn_m': -2,
    'expn_sigma_e': -2,
    'r': 4.08,
    'mu_L': 6.8,
    'sigma_L': 2.2,
    'q': 0.5,
    'food_scheme': 'increasing', 
    'mate_pref': False,
    'h_max': 1,
    'd': 0.68,
    'mate_pickiness': 0.2,
    'low_memory_mode': False
}

parms['nat_death_prob'] = {
    'female': parms['d']*np.array([0, 0.17, 0.16, 0.5, 0.25, 0.5, 0.17, 0.40, 0.40, 1/parms['d']]),
    'male': parms['d']*np.array([0, 0.18, 0.36, 0.19, 0.45, 0.5, 0.5, 0.33, 0.33, 1/parms['d']])
}

# if this verification has already been simulated, just load the file, otherwise, run it and save it
if os.path.isfile("data/verification/forced_dogs.pickle"):
    with open("data/verification/forced_dogs.pickle", "rb") as f:
        x = pickle.load(f)
    forced_dogs_system = x['system']
    forced_dogs_census = x['census']
else:
    forced_dogs_system = evolution_system.system(parms = parms, name = 'forced_dogs')
    forced_dogs_census = forced_dogs_system.simulate()
    while len(forced_dogs_census) < 1499:
        forced_dogs_system = evolution_system.system(parms = parms, name = 'forced_dogs')
        forced_dogs_census = forced_dogs_system.simulate()
    with open("data/verification/forced_dogs.pickle", "wb") as f:
        pickle.dump({'system': forced_dogs_system, 'census': forced_dogs_census}, f)

fig, ax = plt.subplots()
years = [0, 500, 1000, 1499]
years_labels = ['30,000 YBP', '25,000 YBP', '20,000 YBP', '15,000 YBP']
line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
for year, lab, ls  in zip(years, years_labels, line_styles):
    tau_values = forced_dogs_census[year]['taus']
    sns.kdeplot(tau_values, ax = ax, label = lab, cut = 0, clip = (0,1), linestyle = ls, linewidth = 2, color = 'black')

ax.set_xlabel('Human Tolerance, ' + r'$\tau$', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.legend(loc='best', fontsize=10)
fig.savefig('figures/verification/forced_dogs.pdf')

## second system, two populations - set h_max at 50% constantly and turn on mate preference
## and begin with one population at tau = 0.5 and show it splits into two groups.
parms = {
    'c': 500,
    'mu_tau': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'sigma_tau': 2,
    'expn_sigma_f': 0.3,
    'expn_m': -2,
    'expn_sigma_e': -2,
    'r': 4.08,
    'mu_L': 6.8,
    'sigma_L': 2.2,
    'q': 0.5,
    'food_scheme': 'constant', 
    'mate_pref': True,
    'h_max': 1,
    'd': 0.68,
    'mate_pickiness': 0.2,
    'low_memory_mode': False
}

parms['nat_death_prob'] = {
    'female': parms['d']*np.array([0, 0.17, 0.16, 0.5, 0.25, 0.5, 0.17, 0.40, 0.40, 1/parms['d']]),
    'male': parms['d']*np.array([0, 0.18, 0.36, 0.19, 0.45, 0.5, 0.5, 0.33, 0.33, 1/parms['d']])
}

# if this verification has already been simulated, just load the file, otherwise, run it and save it
if os.path.isfile("data/verification/two_pops.pickle"):
    with open("data/verification/two_pops.pickle", "rb") as f:
        x = pickle.load(f)
    two_pops_system = x['system']
    two_pops_census = x['census']
else:
    two_pops_system = evolution_system.system(parms = parms, name = 'two_pops')
    two_pops_census = two_pops_system.simulate()
    while len(two_pops_census) < 1499:
        two_pops_system = evolution_system.system(parms = parms, name = 'two_pops')
        two_pops_census = two_pops_system.simulate()
    with open("data/verification/two_pops.pickle", "wb") as f:
        pickle.dump({'system': two_pops_system, 'census': two_pops_census}, f)

fig, ax = plt.subplots()
years = [0, 500, 1000, 1499]
years_labels = ['30,000 YBP', '25,000 YBP', '20,000 YBP', '15,000 YBP']
line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
for year, lab, ls  in zip(years, years_labels, line_styles):
    tau_values = two_pops_census[year]['taus']
    sns.kdeplot(tau_values, ax = ax, label = lab, cut = 0, clip = (0,1), linestyle = ls, linewidth = 2, color = 'black')

ax.set_xlabel('Human Tolerance, ' + r'$\tau$', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.legend(loc='best', fontsize=10)
fig.savefig('figures/verification/two_pops.pdf')


## thid system, generalists - set h_max at 50% constantly and turn off mate preference
## and begin with one population at tau = 0.5 and show it doesn't form two groups.
parms = {
    'c': 500,
    'mu_tau': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'sigma_tau': 2,
    'expn_sigma_f': 0.3,
    'expn_m': -2,
    'expn_sigma_e': -2,
    'r': 4.08,
    'mu_L': 6.8,
    'sigma_L': 2.2,
    'q': 0.5,
    'food_scheme': 'constant', 
    'mate_pref': False,
    'h_max': 1,
    'd': 0.68,
    'mate_pickiness': 0.2,
    'low_memory_mode': False
}

parms['nat_death_prob'] = {
    'female': parms['d']*np.array([0, 0.17, 0.16, 0.5, 0.25, 0.5, 0.17, 0.40, 0.40, 1/parms['d']]),
    'male': parms['d']*np.array([0, 0.18, 0.36, 0.19, 0.45, 0.5, 0.5, 0.33, 0.33, 1/parms['d']])
}

# if this verification has already been simulated, just load the file, otherwise, run it and save it
if os.path.isfile("data/verification/generalists.pickle"):
    with open("data/verification/generalists.pickle", "rb") as f:
        x = pickle.load(f)
    generalists_system = x['system']
    generalists_census = x['census']
else:
    generalists_system = evolution_system.system(parms = parms, name = 'generalists')
    generalists_census = generalists_system.simulate()
    while len(two_pops_census) < 1499:
        generalists_system = evolution_system.system(parms = parms, name = 'generalists')
        two_pops_census = generalists_system.simulate()
    with open("data/verification/generalists.pickle", "wb") as f:
        pickle.dump({'system': generalists_system, 'census': generalists_census}, f)

fig, ax = plt.subplots()
years = [0, 500, 1000, 1499]
years_labels = ['30,000 YBP', '25,000 YBP', '20,000 YBP', '15,000 YBP']
line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
for year, lab, ls  in zip(years, years_labels, line_styles):
    tau_values = generalists_census[year]['taus']
    sns.kdeplot(tau_values, ax = ax, label = lab, cut = 0, clip = (0,1), linestyle = ls, linewidth = 2, color = 'black')

ax.set_xlabel('Human Tolerance, ' + r'$\tau$', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.legend(loc='best', fontsize=10)
fig.savefig('figures/verification/generalists.pdf')