import evolution_system
import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib
import matplotlib.colors as mcolors
import seaborn as sns

### Simualations
with open("data/verification/generalists.pickle", "rb") as f:
    x = pickle.load(f)
    verification_system_1 = x['system']
    census = x['census']

fig, ax = plt.subplots(figsize = (18, 10))

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

cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Relative Density', fontsize=30, labelpad=20)
cbar.ax.tick_params(labelsize=18)


ax.set_ylim(0, 1)

ax.set_xlim(30000, 15000)

ax2 = ax.twinx()
ax2.plot([data["YBP"] for data in census], [data["human_food"] for data in census], linewidth = 3, label = 'Human Food', color='orange', linestyle='dashed')
ax2.plot([data["YBP"] for data in census], [data["wild_food"] for data in census], linewidth = 3, label = 'Wild Food', color='purple', linestyle='dashed')
#ax2.legend(fontsize=10)
ax2.set_ylim(0, x['system'].parms['c'])

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles = handles1 + handles2
labels = labels1 + labels2
ax.legend(handles, labels, fontsize=20, loc = "upper left")
ax.tick_params(axis='both', labelsize=18)
ax2.tick_params(axis='both', labelsize=18)

ax.set_ylabel("Human Tolerance, " +  r"$\tau$", fontsize=30, labelpad=20)
ax2.set_ylabel("Food Quantity", fontsize=30, labelpad=20)
ax.set_xlabel("Years Before Present (YBP)", fontsize=30, labelpad=20)
fig.savefig('figures/verification/generalists.pdf')