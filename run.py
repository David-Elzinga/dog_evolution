import simulator
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib
import matplotlib.colors as mcolors
import seaborn as sns

p = {
    'seed': 0,
    'c': 500,
    'mu_tau': [0.1],
    'sigma_tau': 0.05,
    'expn_sigma_f': 0.5,
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
    'plot': True
}

p['nat_death_prob'] = {
    'f': p['d']*np.array([0, 0.17, 0.16, 0.5, 0.25, 0.5, 0.17, 0.40, 0.40, 1/p['d']]),
    'm': p['d']*np.array([0, 0.18, 0.36, 0.19, 0.45, 0.5, 0.5, 0.33, 0.33, 1/p['d']])
}

test_system = simulator.system(p, 'test_system')
census = simulator.simulate(test_system)

fig, ax = plt.subplots()

# Define a list of colors and their positions
colors = [(1, 1, 1), (0.7, 0.7, 0.7), (0.4, 0.4, 0.4), (0, 0, 0)]
positions = [0, 0.05, 0.25, 1]

# Create a colormap with modified values
newcmap = mcolors.LinearSegmentedColormap.from_list('customized', list(zip(positions, colors)))
legend_flag=True
for data in census:
    year = data["YBP"]
    tau_values = data["taus"]
    kde = gaussian_kde(tau_values)
    x_vals = np.linspace(0, 1, num=100)
    y_vals = kde(x_vals)
    y_vals_norm = (y_vals - np.min(y_vals)) / (np.max(y_vals) - np.min(y_vals))
    if year % 10 == 0: 
        sc = ax.scatter([year] * len(x_vals), x_vals, c=y_vals_norm, cmap=newcmap, linewidth=2)
        if data["dip_p"] < 0.05: 
            if legend_flag:
                ax.scatter([year], [0], c='red', s=50, zorder = 999, label=r'Dip Test $p < 0.05$')
                legend_flag = False
            else:
                ax.scatter([year], [0], c='red', s=50, zorder = 999)

cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Relative Density', fontsize=14)

ax.set_ylim(0, 1)

ax.set_xlim(30000, 15000)

ax2 = ax.twinx()
ax2.plot([data["YBP"] for data in census], [data["human_food"] for data in census], label = 'Human Food', color='orange', linestyle='dashed')
ax2.plot([data["YBP"] for data in census], [data["wild_food"] for data in census], label = 'Wild Food', color='purple', linestyle='dashed')
#ax2.legend(fontsize=10)
ax2.set_ylim(0, p['c'])

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles = handles1 + handles2
labels = labels1 + labels2
ax.legend(handles, labels, fontsize=10)

ax.set_ylabel("Human Tolerance, " +  r"$\tau$", fontsize=14)
ax2.set_ylabel("Food Quantity", fontsize=14)
ax.set_xlabel("Years Before Present (YBP)", fontsize=14)
plt.show()