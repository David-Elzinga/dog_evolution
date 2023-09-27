import simulator
import numpy as np
import pickle
import os
import pandas as pd
import glob
from itertools import product
from multiprocessing import Pool
import argparse

from collections import Counter


import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib as mpl
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.patches as mpatches


default_cores = os.cpu_count() - 1
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--ncores", type=int, default = default_cores,
                    help="number of cores, defaults to {} on this machine".format(default_cores))
parser.add_argument("-m", "--mode", type=str, default = 'generate',
                    help="generate or plot")

def worker(param_values):

    # Define the dictionary of parameters 
    p = {
        'seed': int(param_values[2]),
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
        'food_scheme': param_values[1],  # if constant, half the value
        'mate_pref': bool(param_values[0]),
        'h_max': 0.3,
        'd': 0.68
    }

    p['nat_death_prob'] = {
        'f': p['d']*np.array([0, 0.17, 0.16, 0.5, 0.25, 0.5, 0.17, 0.40, 0.40, 1/p['d']]),
        'm': p['d']*np.array([0, 0.18, 0.36, 0.19, 0.45, 0.5, 0.5, 0.33, 0.33, 1/p['d']])
    }

    census = []
    num_tries = 0
    while len(census) < 15000 and num_tries < 50:
        validation_system = simulator.system(p, param_values[3])
        census = simulator.simulate(validation_system)
        num_tries += 1

    with open("data/validation/" + param_values[3], "wb") as f:
        pickle.dump({'system': validation_system, 'census': census, 'num_tries': num_tries}, f)

def generate(ncores = None, pool = None):

    # Create a dataframe holding all possible combinations of food scheme, mate preference, and repetition number/seed
    df = pd.DataFrame(list(product([str(True), str(False)], ['constant', 'increasing'], [str(i) for i in range(5)])), columns=['mate_pref', 'food_scheme', 'rep_seed'])
    df['filename'] = 'validation_matepref-' + df['mate_pref'] + '_foodscheme-' + df['food_scheme'] + '_rep-' + df['rep_seed'] + '.pickle'
    completed_files = glob.glob('data/validation/*')
    df['completed'] = df['filename'].isin(completed_files)

    # Run all the files whose completed value is false. 
    tbc_df = df[~df['completed']]
    print('running...')
    pool.map(worker, tbc_df.values)

    return

def plot():

    # Iterate over the four combinations of mate preference / human food
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(15,10))  

    for col, mate_pref in enumerate([False, True]):
        for row, food_scheme in enumerate(['constant', 'increasing']):

            # Load in all the files that have this combination
            files = glob.glob('data/validation/validation_matepref-' + str(mate_pref) + '_foodscheme-' + food_scheme + '*.pickle') 
            final_age_props = []
            for n, f in enumerate(files):
                with open("data/validation/" + f.split('\\')[1], "rb") as x:
                    print('Reading in file ' + str(n) + ' out of ' + str(len(files)))
                    p = pickle.load(x)

                    final_ages = p['census'][-1]['ages']
                    counts = Counter(final_ages)
                    total = len(final_ages)
                    age_props = {}

                    for i in range(8):
                        age_props[i] = counts[i] / total

                    age_props[8] = (counts[8] + counts[9]) / total
                    
                    final_age_props.append(age_props)

            plot_data = {}  
            for i in range(9):
                plot_data[i] = [d[i] for d in final_age_props]  

            violin_parts = ax[row, col].violinplot(plot_data.values(), positions=range(9))
            ax[row, col].set_xticks(range(9))
            ax[row, col].set_xticklabels(['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9+'], rotation=45)
            ax[row, col].tick_params(axis='x', labelsize=14)
            ax[row, col].tick_params(axis='y', labelsize=14)


            # for partname in ('cbars','cmins','cmaxes'):
            #     vp = violin_parts[partname]
            #     vp.set_edgecolor("black")
            #     vp.set_linewidth(1)

            # for vp in violin_parts['bodies']:
            #     vp.set_facecolor("deepskyblue")
            #     vp.set_edgecolor("black")
            #     vp.set_linewidth(1)
            #     vp.set_alpha(0.5)

            ax[row, col].scatter([0, 1, 2, 3, 4, 5, 6, 7, 8], [0.34, 0.21, 0.15, 0.11, 0.05, 0.05, 0.04, 0.02, sum([0.02, 0.01])], facecolors="blueviolet", edgecolors='black', marker='d', s=100, label="Empirical Data", zorder=300)
            
            if row == 0 and col == 0:
                green_patch = mpatches.Patch(color="lightskyblue")
                ax[row, col].legend(handles=[green_patch, *ax[row, col].get_legend_handles_labels()[0]], labels=['Simulation', *ax[row, col].get_legend_handles_labels()[1]], fontsize=12)

    ax[1,0].set_xlabel('Ages', fontsize=18, labelpad=10); ax[1,1].set_xlabel('Ages', fontsize=18, labelpad=10)
    ax[0,0].set_ylabel('Probability', fontsize=18, labelpad=10); ax[1,0].set_ylabel('Probability', fontsize=18, labelpad=10)
    fig.subplots_adjust(hspace=0.15)
    
    cols = ['No Mate Pref.', 'Mate Pref.']
    rows = ['H.F. Constant', 'H.F. Increasing']

    pad = 5

    for a, col in zip(ax[0], cols):
        a.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    fontsize=20, ha='center', va='baseline')
    
    for a, row in zip(ax[:,0], rows):
        a.annotate(row, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - pad, 0),
                    xycoords=a.yaxis.label, textcoords='offset points',
                    fontsize=20, ha='right', va='center')

    fig.tight_layout()
    fig.subplots_adjust(left=0.21, top=0.95)
    fig.savefig('figures/validation/ages_validation.pdf')




    return




if __name__ == "__main__":
    args = parser.parse_args()

    if args.mode == 'generate':
        with Pool(args.ncores) as pool:
            generate(args.ncores, pool)
    elif args.mode == 'plot':
        plot()


















# Give random seeds


# # Compete 10 repetitions of each system
# validation_systems = []
# censuses = []
# for mate_pref in [True, False]:
#     for food_scheme in ['increasing', 'constant']:
#         for s in range(10):
#             if os.path.isfile("data/validation/validation_matepref-" + str(mate_pref) + "_foodscheme-" + food_scheme + "_rep-" + str(s) + ".pickle"):
#                 with open("data/validation/validation_matepref-" + str(mate_pref) + "_foodscheme-" + food_scheme + "_rep-" + str(s) + ".pickle", "rb") as f:
#                     x = pickle.load(f)
#                 validation_systems.append(x['system'])
#                 censuses.append(x['census'])

#             else:
#                 p['seed'] = s; p['mate_pref'] = mate_pref; p['food_scheme'] = food_scheme
#                 validation_system = simulator.system(p, 'validation system')
#                 census = simulator.simulate(validation_system)
#                 censuses.append(census)
#                 with open("data/validation/validation_matepref-" + str(mate_pref) + "_foodscheme-" + food_scheme + "_rep-" + str(s) + ".pickle", "wb") as f:
#                     pickle.dump({'system':validation_system, 'census':census}, f)

# # For each of the validation systems, collect ages over the past 10 decades 
# ages_by_system = []
# for s in range(10):
#     ages = []
#     for decade in range(-1, -11, -1):
#         ages += censuses[s][decade]['ages']

#     counts = Counter(ages)
#     total = len(ages)
#     age_props = {}

#     for i in range(0, 10):
#         age_props[i] = counts[i] / total  

#     ages_by_system.append(age_props)


# data = {}  
# for i in range(10):
#     data[i] = [d[i] for d in ages_by_system]  

# fig, ax = plt.subplots()  
# ax.violinplot(data.values(), positions=range(10))
# ax.scatter([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.34, 0.21, 0.15, 0.11, 0.05, 0.05, 0.04, 0.02, 0.02, 0.01], color='red', marker='d')

# # ax.set_xticklabels(data.keys()) 
# # ax.set_xlabel("Integer Keys")
# # ax.set_ylabel("Values")
# # ax.set_title("Box and Whisker Plot for Integer Keys")

# plt.show()  
