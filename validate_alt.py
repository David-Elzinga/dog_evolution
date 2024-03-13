import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
import evolution_system
from itertools import product
from collections import Counter
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib.patches as mpatches

default_cores = os.cpu_count() - 1
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--ncores", type = int, default = default_cores,
                    help = "number of cores, defaults to {} on this machine".format(default_cores))
parser.add_argument("-m", "--mode", type = str, 
                    help = "generate: generates data for validation, or plot: to plot validation figure")

def worker(non_default_parms):

    # define the parameter values
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
        'food_scheme': non_default_parms[0], 
        'mate_pref': bool(non_default_parms[1]),
        'h_max': 0.3,
        'd': 0.68,
        'mate_pickiness': 0.2,
        'low_memory_mode': False
    }

    parms['nat_death_prob'] = {
        'female': parms['d']*np.array([0, 0.17, 0.16, 0.5, 0.25, 0.5, 0.17, 0.40, 0.40, 1/parms['d']]),
        'male': parms['d']*np.array([0, 0.18, 0.36, 0.19, 0.45, 0.5, 0.5, 0.33, 0.33, 1/parms['d']])
    }

    # for 50 attempts, simulate the validation system and pickle the results
    validation_census = []; num_runs = 0
    while len(validation_census) < 1500 and num_runs < 50:
        validation_system = evolution_system.system(parms = parms, name = 'validation_system')
        validation_census = validation_system.simulate()
        num_runs += 1

    with open("data/validation/" + non_default_parms[3], "wb") as f:
        pickle.dump({'system': validation_system, 'census': validation_census, 'num_runs': num_runs}, f)

def generate(ncores = None, pool = None):

    # create a df holding all possible combinations of food scheme, mate preference, and repetition number
    df = pd.DataFrame(list(product(['constant', 'increasing'], [str(True), str(False)], [str(i) for i in range(5)])), columns=['food_scheme', 'mate_pref', 'rep'])
    df['filename'] = 'validation_foodscheme-' + df['food_scheme'] + '_matepref-' + df['mate_pref'] + '_rep-' + df['rep'] + '.pickle'
    completed_files = glob.glob('data/validation/*')
    df['completed'] = df['filename'].isin(completed_files)
    
    # run all the files whose completed value is false.
    tbc_df = df[~df['completed']]
    print('running...')
    pool.map(worker, tbc_df.values)
    return

def plot():

    # create 2x2 fig for the two food schemes and mate preferences
    fig, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (15,10))  
    for col, mate_pref in enumerate([False, True]):
        for row, food_scheme in enumerate(['constant', 'increasing']):

            # load in all the files that have this combination
            files = glob.glob('data/validation/validation_foodscheme-' + food_scheme + '_matepref-' + str(mate_pref) + '*.pickle') 
           
            # collect the proportion of canines of each age in the final census reading of each file
            final_age_props = []
            for n, f in enumerate(files):
                with open("data/validation/" + f.split('\\')[1], "rb") as x:
                    print('Reading in file ' + str(n) + ' out of ' + str(len(files)))
                    p = pickle.load(x)
                    final_ages = np.array(p['census'][-1]['ages']) + 1 # wolves really represent one further year
                    counts = Counter(final_ages)
                    total = len(final_ages)
                    age_props = {}
                    for i in range(1, 10):
                        age_props[i] = counts[i] / total
                    final_age_props.append(age_props)

            # group the age data by age instead of by repetition
            plot_data = {}  
            for i in range(1, 10):
                if i == 1:
                    plot_data[i] = [d[1] + d[2] for d in final_age_props]  # group together the first two ages
                elif i == 2:
                    pass
                else:
                    plot_data[i] = [d[i] for d in final_age_props]  
            plot_data = list(plot_data.values()) 

            # make violin plots for the simulation data
            violin_parts = ax[row, col].violinplot(plot_data, positions=range(8))
            ax[row, col].set_xticks(range(0, 8))
            ax[row, col].set_xticklabels(['1 & 2', '3', '4', '5', '6', '7', '8', '9'], rotation=45)
            ax[row, col].tick_params(axis='x', labelsize=14)
            ax[row, col].tick_params(axis='y', labelsize=14)
            
            # add a purple diamond to represent the empirical data (combining the last two age groups)
            ax[row, col].scatter([0, 1, 2, 3, 4, 5, 6, 7], [0.52, 0.15, 0.15, 0.09, 0.03, 0.03, 0.01, 0.04], facecolors="blueviolet", edgecolors='black', marker='d', s=100, label="Empirical Data", zorder=300)
            # manually construct the legend
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
    fig.subplots_adjust(left=0.28, top=0.95)

    plt.show()
    fig.savefig('figures/validation/ages_validation.pdf')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.mode == 'generate':
        with Pool(args.ncores) as pool:
            generate(args.ncores, pool)
    elif args.mode == 'plot':
        plot()