import os
import glob
import pickle
import argparse
import itertools
import traceback
import numpy as np
import pandas as pd
import evolution_system
from multiprocessing import Pool

from SALib.analyze import fast
from SALib.sample import fast_sampler

default_cores = os.cpu_count() - 1
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--ncores", type = int, default = default_cores,
                    help = "number of cores, defaults to {} on this machine".format(default_cores))
parser.add_argument("-m", "--mode", type = str, 
                    help = "generate: generates data for validation, or plot: to plot validation figure")

def worker(parm_values):

    # define the parameter values
    parms = {
        'c': int(parm_values[0]),
        'mu_tau': [parm_values[1]],
        'sigma_tau': 0.2,
        'expn_sigma_f': parm_values[2],
        'expn_m': parm_values[3],
        'expn_sigma_e': parm_values[4],
        'r': parm_values[5],
        'mu_L': parm_values[6],
        'sigma_L': parm_values[7],
        'q': parm_values[8],
        'h_max': parm_values[9],
        'd': parm_values[10],
        'mate_pref': bool(parm_values[12]), # skip over the dummy parameter
        'food_scheme': parm_values[13],
        'low_memory_mode': True
    }
    parms['nat_death_prob'] = {
        'female': parms['d']*np.array([0, 0.17, 0.16, 0.5, 0.25, 0.5, 0.17, 0.40, 0.40, 1/parms['d']]),
        'male': parms['d']*np.array([0, 0.18, 0.36, 0.19, 0.45, 0.5, 0.5, 0.33, 0.33, 1/parms['d']])
    }

    # for 50 attempts, simulate the efast system and pickle the results
    efast_census = []; num_runs = 0
    while len(efast_census) < 1500 and num_runs < 50:
        efast_system = evolution_system.system(parms = parms, name = 'efast_system')
        try:
            efast_census = efast_system.simulate()
        except Exception as e:
            efast_census = 'error'
            error_details = {
                'type': type(e),
                'args': e.args,
                'message': str(e),
                'traceback': traceback.format_exc()
            }
            with open('error_details.pickle', 'wb') as file:
                pickle.dump(error_details, file)
            break
        num_runs += 1

    with open("data/efast/" + parm_values[-1], "wb") as f:
        pickle.dump({'system': efast_system, 'census': efast_census, 'num_runs': num_runs}, f)

def generate(ncores = None, pool = None):
    
    # check to see if the master df (which organizes all simulations) has been created - if 
    # so, read it in, otherwise, create it
    if os.path.isfile("data/efast/master_df.pickle"):
        with open("data/efast/master_df.pickle", "rb") as f:
            master_df = pickle.load(f)
    else:
        problem = {
            'num_vars': 12,
            'names': ['c', 'mu_tau', 'expn_sigma_f', 'expn_m', 'expn_sigma_e', 'r', 'mu_L', 'sigma_L', 'q', 'h_max', 'd', 'dummy'],
            'bounds': [
                [400, 600],             # Range for 'c'
                [0, 0.5],               # Range for 'mu_tau'
                [-2, 0.5],              # Range for 'expn_sigma_f'
                [-3, -1],               # Range for 'expn_m'
                [-3, -1],               # Range for 'expn_sigma_e'
                [0.8*4.08, 1.2*4.08],   # Range for 'r'
                [0.8*6.8, 1.2*6.8],     # Range for 'mu_L'
                [0.8*2.2, 1.2*2.2],     # Range for 'sigma_L'
                [0.4, 0.6],             # Range for 'q'
                [0.01, 0.5],            # Range for 'h_max'
                [0.8*0.68, 1.2*0.68],   # Range for 'd'
                [0, 1]                  # Range for 'dummy'
            ]
        }

        efast_params = fast_sampler.sample(problem, N = 65*12, seed = 0)

        # generate all combinations of mate_pref, food_scheme, and rep
        mate_food_rep_combinations = list(itertools.product([True, False], ['constant', 'increasing'], range(10)))

        # create a list to hold all the parameter sets
        parameter_sets = []

        # Iterate over each parameter and generate the simulations
        for n, values in enumerate(efast_params):
            for mate_pref, food_scheme, rep in mate_food_rep_combinations:
                parm_variation = dict(zip(problem['names'], values))
                parm_variation["mate_pref"] = mate_pref
                parm_variation["food_scheme"] = food_scheme
                parm_variation["rep"] = rep
                parm_variation['param_set'] = n
                parameter_sets.append(parm_variation)

        # create a DataFrame from the list of parameter sets
        master_df = pd.DataFrame(parameter_sets)

        # reset index and create file names
        master_df.reset_index(drop=True, inplace=True)
        master_df['filename'] = 'efast-' + master_df['param_set'].astype(str).values + '_matepref-' + master_df['mate_pref'].astype(str).values + '_foodscheme-' + master_df['food_scheme'] + '_rep-' + master_df['rep'].astype(str).values + '.pickle'
        with open("data/efast/master_df.pickle", "wb") as f:
            pickle.dump(master_df, f)

    # identify the completed files
    completed_files = [f.split('\\')[-1] for f in glob.glob('data/efast/*')]
    master_df['completed'] = master_df['filename'].isin(completed_files)
    
    # run all the files whose completed value is false. 
    tbc_df = master_df[~master_df['completed']]
    tbc_df.drop('completed', axis=1, inplace=True)
    print('running...')
    print(tbc_df.shape)
    pool.map(worker, tbc_df.values)


def parse():

    # read in the master df 
    with open("data/efast/master_df.pickle", "rb") as x:
        df = pickle.load(x)

    df['num_runs'] = np.nan
    df['max_streak'] = np.nan
    df['speciation_ybp'] = np.nan

    # identify each of the complete files
    completed_files = glob.glob('data/efast/efast*')

    # open each file, parse the results, put them in the master df
    for n, f in enumerate(completed_files):
        print('Working on file ' + str(n) + ' out of ' + str(len(completed_files)))
        with open(f, "rb") as x:
            results = pickle.load(x)

        filename = f.split(os.sep)[-1]

        # record the number of tries
        df.loc[df['filename'] == filename, 'num_runs'] = results['num_runs']

        # check for an error, record if there was/wasn't an error
        if results['census'] == 'error':
            df.loc[df['filename'] == filename, 'error'] = True

        else:
            df.loc[df['filename'] == filename, 'error'] = False
            
            # collect the p-values as being significant/not significant
            p_vals = [1*(x['dip_p'] < 0.1) for x in results['census']]

            # check if the number of p-values is at least 1500
            max_len = 0; cur_len = 0
            for num in p_vals:
                if num == 1:
                    cur_len += 1
                else:
                    max_len = max(max_len, cur_len)
                    cur_len = 0
            df.loc[df['filename'] == filename, 'max_streak'] = max(max_len, cur_len)

            # now we look for the first occurance of speciation (at least 15 consecutive decades
            # where the p-value is < 0.05) Convolve over the p_vals with an array of 15 ones,
            # look for the sum to be 15 (this indicates consecutive decades of speciation)
            convolution = np.convolve(p_vals, np.ones(15), mode='valid') == 15
            if convolution.any(): # check if there's a place where we achieve the 15
                index_of_spec = np.argmax(convolution)
            else: # otherwise return -1 as an indicator there's no speciation
                index_of_spec = -1

            # if there is no speciation event - record speciation generation at nan
            if index_of_spec == -1:
                df.loc[df['filename'] == filename, 'speciation_ybp'] = np.nan
            else: # If there is - check if the index is zero (meaning speciation started immediately)
                if index_of_spec == 0:
                    df.loc[df['filename'] == filename, 'speciation_ybp'] = 30000
                else: # assuming the speciation exists - look into the "starts" list to record where it begins
                    df.loc[df['filename'] == filename, 'speciation_ybp'] = 30000 - 10*(index_of_spec + 1)

        with open("data/efast/master_df_parsed.pickle", "wb") as x:
            pickle.dump(df, x)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.mode == 'generate':
        with Pool(args.ncores) as pool:
            generate(args.ncores, pool)
    elif args.mode == 'parse':
        parse()
    elif args.mode == 'plot':
        plot()