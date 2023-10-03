# %% Import packages
import pandas as pd
import numpy as np
from eztao.carma import CARMA_term

# %% Function to extract parameter names
def parameter_names(m,p,q):
    para_names = []
    for i in range(p+q+1):
        para_names.append("carma"+str(i+1))
    para_names.append("theta0")
    for j in range(m):
        para_names.append("theta"+str(j+1))
    para_names.append("delta")
    return para_names

def column_names(para_names):
    colnames = []
    for par in para_names:
        colnames.append(par+'_mean')
    for par in para_names:
        colnames.append(par+'_std')
    for par in para_names:
        colnames.append(par+'_bestfit')
    for par in para_names:
        colnames.append(par+'_map')
    colnames.append('ln(Z)')
    colnames.append('max_ll')
    return colnames

def model_name(m,p,q):
    if p == 1:
        return 'TD-DRW('+str(m)+')'
    else:
        return 'TD-CARMA('+str(p)+','+str(q)+','+str(m)+')'

def lorentzian(ar_paras):
    roots = np.roots(ar_paras)
    lorentz_cent = np.abs(roots.imag) / (2.0 * np.pi)
    lorentz_width = - 1.0  * roots.real / (2.0 * np.pi)
    return lorentz_cent, lorentz_width

def check_frequency(df, p, q):

    log_fcoeff = [df['carma'+str(i)+'_mean'] for i in range(1, p+q+2)]
    log_ar, _ = CARMA_term.fcoeffs2carma_log(log_fcoeff, p)
    ar_coefs = np.concatenate([[1], np.exp(log_ar)])

    lorentz_width = lorentzian(ar_coefs)[0]

    return np.unique(lorentz_width)

# %% Process results
def process_results_file(output_file, m, p, q):

    # Prepare column names for dataframe
    para_names = parameter_names(m,p,q)
    col_names = column_names(para_names)

    # Export .txt results file as pandas
    file_df = pd.read_csv(output_file, sep = "   ", header = None, names = col_names, engine='python')

    # Row names 
    mode_rows = ['Full'] + ['Mode'+str(i) for i in range(1, len(file_df))]
    file_df.insert(0, 'Mode', mode_rows)

    # Model names
    modelname = model_name(m,p,q)
    model_rows = [modelname for i in range(len(file_df))]
    file_df.insert(0, 'Model', model_rows)

    # Check if there is a frequency
    freqs = []
    for i in range(len(file_df)):
        freqs.append(check_frequency(file_df.loc[i], p, q))

    file_df.insert(len(file_df.columns)-2, 'Freq', freqs)

    # Select relevant columns
    selected_columns = ['Model', 'Mode', 'delta_mean', 'delta_std','Freq', 'ln(Z)']
    final_df = file_df[selected_columns]

    if len(mode_rows) == 2:
        final_df = final_df.drop(1)

    return final_df
