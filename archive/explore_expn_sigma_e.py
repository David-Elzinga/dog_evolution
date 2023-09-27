import simulator
import numpy as np
import pickle
import os
import pandas as pd
import glob
from itertools import product
from multiprocessing import Pool
import argparse
import traceback

from collections import Counter
from SALib.sample.latin import sample



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
        'c': int(param_values[0]),
        'mu_tau': [param_values[1]],
        'sigma_tau': 0.2,
        'expn_sigma_f': param_values[2],
        'expn_m': param_values[3],
        'expn_sigma_e': param_values[4],
        'r': param_values[5],
        'mu_L': param_values[6],
        'sigma_L': param_values[7],
        'q': param_values[8],
        'h_max': param_values[9],
        'd': param_values[10],
        'mate_pref': bool(param_values[13]),
        'food_scheme': param_values[14],  # if constant, half the value
        'seed': np.random.randint(9999999),
    }

    p['nat_death_prob'] = {
        'f': p['d']*np.array([0, 0.17, 0.16, 0.5, 0.25, 0.5, 0.17, 0.40, 0.40, 1/p['d']]),
        'm': p['d']*np.array([0, 0.18, 0.36, 0.19, 0.45, 0.5, 0.5, 0.33, 0.33, 1/p['d']])
    }

    census = []
    num_tries = 0
    while len(census) < 1500 and num_tries < 50:  
        #print(num_tries)
        system = simulator.system(p, param_values[-1])
        try:
            census = simulator.simulate(system)

        except Exception as e:
            census = 'Error' 

            error_details = {
                'type': type(e),
                'args': e.args,
                'message': str(e),
                'traceback': traceback.format_exc()
            }
            with open('error_details.pickle', 'wb') as file:
                pickle.dump(error_details, file)
            break

        num_tries += 1; p['seed'] = np.random.randint(9999999)

    with open("data/explore_expn_sigma_e/" + param_values[-1], "wb") as f:
        pickle.dump({'system': system, 'census': census, 'num_tries': num_tries}, f)

def generate(ncores = None, pool = None):

    # Check to see if the master df is in the directory. If so, read it in. Otherwise, generate it. 
    if os.path.isfile("data/explore_expn_sigma_e/master_df.pickle"):
        with open("data/explore_expn_sigma_e/master_df.pickle", "rb") as x:
            df = pickle.load(x)
            print('found an existing master file')
    else:
        # Define a problem for SALib
        problem = {
            'num_vars': 12,
            'names': ['c', 'mu_tau', 'expn_sigma_f', 'expn_m', 'expn_sigma_e', 'r', 'mu_L', 'sigma_L', 'q', 'h_max', 'd', 'dummy'],
            'bounds': [[400, 600], # c
                       [0, 0.5], # mu_tau
                       [-2, 0.5], # expn_sigma_f
                       [-4, -1], # expn_m
                       [-3, -1], # expn_sigma_e
                       [0.8*4.08, 1.2*4.08], # r
                       [0.8*6.8, 1.2*6.8], # mu_L
                       [0.8*2.2, 1.2*2.2], # sigma_L
                       [0.4, 0.6], # q
                       [0.01, 0.5], # h_max
                       [0.8*0.68, 1.2*0.68], # d
                       [0, 1]] # dummy
        }

        # Draw our LHS samples
        lhs_samples = sample(problem, 60)

        # Create a DataFrame from the list of parameter combinations
        df = pd.DataFrame(lhs_samples, columns = problem['names'])
        df['param_set_id'] = df.index 

        # Now we need to replicate this list of parameter combinations for each combination 
        # of mate preference/food_scheme and across 50 repetitions. 
        mate_pref = [True, False]
        food_scheme = ['constant', 'increasing']
        rep = range(50)
        df_crossed = pd.DataFrame(list(product(mate_pref, food_scheme, rep)), 
                                columns=['mate_pref', 'food_scheme', 'rep'])

        # Merge our two df's on a dummy key (then drop it.)
        df['key'] = 1
        df_crossed['key'] = 1
        df = pd.merge(df, df_crossed, on='key')
        df.drop('key', axis=1, inplace=True)

        # Construct file names in the following format, then dump it.
        df['filename'] = 'explore_expn_sigma_e-' + df['param_set_id'].astype(str).values + '_matepref-' + df['mate_pref'].astype(str).values + '_foodscheme-' + df['food_scheme'] + '_rep-' + df['rep'].astype(str).values + '.pickle'
        with open("data/explore_expn_sigma_e/master_df.pickle", "wb") as x:
            pickle.dump(df, x)

    # Identify the completed files
    completed_files = [f.split('data/explore_expn_sigma_e/')[-1] for f in glob.glob('data/explore_expn_sigma_e/*')]
    df['completed'] = df['filename'].isin(completed_files)
    print('found ' + str(df['completed'].sum()) + ' already completed' )
    
    # Run all the files whose completed value is false. 
    tbc_df = df[~df['completed']]
    tbc_df =  tbc_df.drop('completed', axis=1)
    print('running ' + str(tbc_df.shape[0]) + ' simulations')
    pool.map(worker, tbc_df.values)

    return

def parse():

    # Read in the master df 
    with open("data/explore_expn_sigma_e/master_df.pickle", "rb") as x:
        df = pickle.load(x)

    df['num_tries'] = np.nan
    df['max_streak'] = np.nan
    df['speciation_ybp'] = np.nan

    # Identify each of the complete files
    completed_files = glob.glob('data/explore_expn_sigma_e/explore_expn_sigma_e*')

    # Open each file, parse the results, put them in the master df
    for n, f in enumerate(completed_files):
        print('Working on file ' + str(n) + ' out of ' + str(len(completed_files)))
        
        with open(f, "rb") as x:
            results = pickle.load(x)

        filename = f.split(os.sep)[-1]

        # Record the number of tries
        df.loc[df['filename'] == filename, 'num_tries'] = results['num_tries']

        # Check for an error, record if there was/wasn't an error
        if results['census'] == 'Error':
            df.loc[df['filename'] == filename, 'error'] = True

        else:
            df.loc[df['filename'] == filename, 'error'] = False
            
            # Collect the p-values as being significant/not significant
            p_vals = [1*(x['dip_p'] < 0.1) for x in results['census']]

            # Check if the number of p-values is at least 1500
            if len(p_vals) == 1500: # the last try was successful
                
                # Identify the longest streak of speciation
                max_len = 0; cur_len = 0
                for num in p_vals:
                    if num == 1:
                        cur_len += 1
                    else:
                        max_len = max(max_len, cur_len)
                        cur_len = 0
                df.loc[df['filename'] == filename, 'max_streak'] = max(max_len, cur_len)

                # Now we look for the first occurance of speciation (at least 15 consecutive decades
                # where the p-value is < 0.05) Convolve over the p_vals with an array of 15 ones,
                # look for the sum to be 15 (this indicates consecutive decades of speciation)
                convolution = np.convolve(p_vals, np.ones(15), mode='valid') == 15
                if convolution.any(): # check if there's a place where we achieve the 15
                    index_of_spec = np.argmax(convolution)
                else: # otherwise return -1 as an indicator there's no speciation
                    index_of_spec = -1

                # If there is no speciation event - record speciation generation at nan
                if index_of_spec == -1:
                    df.loc[df['filename'] == filename, 'speciation_ybp'] = np.nan
                else: # If there is - check if the index is zero (meaning speciation started immediately)
                    if index_of_spec == 0:
                        df.loc[df['filename'] == filename, 'speciation_ybp'] = 30000
                    else: # assuming the speciation exists - look into the "starts" list to record where it begins
                        df.loc[df['filename'] == filename, 'speciation_ybp'] = 30000 - 10*(index_of_spec + 1)

        with open("data/explore_expn_sigma_e/master_df_parsed.pickle", "wb") as x:
            pickle.dump(df, x)

def plot():

    return




if __name__ == "__main__":
    args = parser.parse_args()

    if args.mode == 'generate':
        with Pool(args.ncores) as pool:
            generate(args.ncores, pool)
    elif args.mode == 'parse':
        parse()
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
