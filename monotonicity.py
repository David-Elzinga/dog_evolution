import os
import glob
import pickle
import argparse
import traceback
import numpy as np
import pandas as pd
import evolution_system
from multiprocessing import Pool

default_cores = 30
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
        'sigma_tau': parm_values[2],
        'expn_sigma_f': parm_values[3],
        'expn_m': parm_values[4],
        'expn_sigma_e': parm_values[5],
        'r': parm_values[6],
        'mu_L': parm_values[7],
        'sigma_L': parm_values[8],
        'q': parm_values[9],
        'h_max': parm_values[10],
        'd': parm_values[11],
        'mate_pickiness': parm_values[12],
        'mate_pref': bool(parm_values[14]), # skip over the dummy parameter
        'food_scheme': parm_values[15],
        'low_memory_mode': True
    }
    parms['nat_death_prob'] = {
        'female': parms['d']*np.array([0, 0.17, 0.16, 0.5, 0.25, 0.5, 0.17, 0.40, 0.40, 1/parms['d']]),
        'male': parms['d']*np.array([0, 0.18, 0.36, 0.19, 0.45, 0.5, 0.5, 0.33, 0.33, 1/parms['d']])
    }
    
    # for 50 attempts, simulate the monotonicity system and pickle the results
    mono_census = []; num_runs = 0
    while len(mono_census) < 1500 and num_runs < 50:
        mono_system = evolution_system.system(parms = parms, name = 'mono_system')
        try:
            mono_census = mono_system.simulate()
        except Exception as e:
            mono_census = 'error'
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

    with open("data/monotonicity/" + parm_values[-1], "wb") as f:
        pickle.dump({'system': mono_system, 'census': mono_census, 'num_runs': num_runs}, f)

def generate(ncores = None, pool = None):
    
    # check to see if the master df (which organizes all simulations) has been created - if 
    # so, read it in, otherwise, create it
    if os.path.isfile("data/monotonicity/master_df.pickle"):
        with open("data/monotonicity/master_df.pickle", "rb") as f:
            master_df = pickle.load(f)
    else:
        # define the default parameters and their ranges
        default = {
            'c': 500,
            'mu_tau': 0.1,
            'sigma_tau': 0.2,
            'expn_sigma_f': 0,
            'expn_m': -2,
            'expn_sigma_e': -2,
            'r': 4.08,
            'mu_L': 6.8,
            'sigma_L': 2.2,
            'q': 0.5,
            'h_max': 0.3,
            'd': 0.68,
            'mate_pickiness': 0.2,
            'dummy': 0.5
        }

        ranges = {
            'c': [400, 600],
            'expn_sigma_f': [-2, 0.5],
            'expn_m': [-3, -1],
            'expn_sigma_e': [-3, -1],
            'r': [0.8*4.08, 1.2*4.08],
            'mu_L': [0.8*6.8, 1.2*6.8],
            'sigma_L': [0.8*2.2, 1.2*2.2],
            'q': [0.4, 0.6],
            'h_max': [0.01, 0.5],
            'd': [0.8*0.68, 1.2*0.68],
            'mate_pickiness': [0, 0.4],
            'dummy': [0, 1]
        }

        # create an empty df to store the simulations
        master_df = pd.DataFrame(columns= list(default.keys()) + ['mate_pref', 'food_scheme', 'rep', 'adj_parm', 'adj_step'])

        # iterate over each parameter and generate the simulations
        for parm_name in ranges.keys():
            parm_range = ranges[parm_name]
            parm_vals = np.linspace(parm_range[0], parm_range[1], 5)
            for n, value in enumerate(parm_vals):
                for mate_pref in [True, False]:
                    for food_scheme in ['constant', 'increasing']:
                        for rep in range(50): # 50 reps per simulation
                            # copy the default dict and adjust the one parm value that is changing
                            parm_variation = default.copy()
                            parm_variation[parm_name] = value
                            parm_variation["mate_pref"] = mate_pref
                            parm_variation["food_scheme"] = food_scheme
                            parm_variation["rep"] = rep
                            parm_variation['adj_parm'] = parm_name
                            parm_variation['adj_step'] = n
                            master_df.loc[len(master_df)] = parm_variation

        # reset index and create file names
        master_df.reset_index(drop=True, inplace=True)
        master_df['filename'] = 'mono-' + master_df['adj_parm'].astype(str).values + master_df['adj_step'].astype(str).values + '_matepref-' + master_df['mate_pref'].astype(str).values + '_foodscheme-' + master_df['food_scheme'] + '_rep-' + master_df['rep'].astype(str).values + '.pickle'
        with open("data/monotonicity/master_df.pickle", "wb") as f:
            pickle.dump(master_df, f)

    # identify the completed files
    completed_files = [f.split('\\')[-1] for f in glob.glob('data/monotonicity/*')]
    master_df['completed'] = master_df['filename'].isin(completed_files)
    
    # Run all the files whose completed value is false. 
    tbc_df = master_df[~master_df['completed']]
    tbc_df.drop('completed', axis=1, inplace=True)
    print('running...')
    print(tbc_df.shape)
    pool.map(worker, tbc_df.values)


def parse():

    # read in the master df 
    with open("data/monotonicity/master_df.pickle", "rb") as x:
        df = pickle.load(x)

    df['num_runs'] = np.nan
    df['max_streak'] = np.nan
    df['speciation_ybp'] = np.nan

    # identify each of the complete files
    completed_files = glob.glob('data/monotonicity/mono*')

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
            p_vals = [1*(x['dip_p'] < 0.01) for x in results['census']]

            # check if the number of p-values is at least 1500
            max_len = 0; cur_len = 0
            for num in p_vals:
                if num == 1:
                    cur_len += 1
                else:
                    max_len = max(max_len, cur_len)
                    cur_len = 0
            df.loc[df['filename'] == filename, 'max_streak'] = max(max_len, cur_len)

            # now we look for the first occurance of speciation (at least 150 consecutive decades
            # where the p-value is < 0.01) Convolve over the p_vals with an array of 150 ones,
            # look for the sum to be 150 (this indicates consecutive decades of speciation)
            convolution = np.convolve(p_vals, np.ones(150), mode='valid') == 150
            if convolution.any(): # check if there's a place where we achieve the 150
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

        with open("data/monotonicity/master_df_parsed_150.pickle", "wb") as x:
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