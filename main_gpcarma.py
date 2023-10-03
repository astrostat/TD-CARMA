#%% Import libraries
from __future__ import absolute_import, unicode_literals, print_function
import numpy as np
from pymultinest.solve import solve
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import os
try: os.mkdir('chains')
except OSError: pass
import pickle
import time
import itertools
from multiprocessing import Pool
import tqdm
#from mpi4py import MPI
plt.style.use('tableau-colorblind10')

import process_gp_results

import eztao
import celerite
from celerite import GP
from eztao.ts import carma_log_fcoeff_init
from eztao.carma import CARMA_term
from eztao.ts import neg_fcoeff_ll

#%% Choose and import data
data_dir = './td_carma/data/'
data_file = '2M1134_WFI.csv'
data = pd.read_csv(data_dir+data_file)

# %% Function that fits TD-CARMA to dataset for given (p,q,m) amd outputs processed table
def fit_tdcarma_gp_(p, q, m, data = data):

    # Extract data
    t = data.iloc[:,0] - min(data.iloc[:,0])
    y = data.iloc[:,1]
    z = data.iloc[:,3]
    yerr = data.iloc[:,2]
    zerr = data.iloc[:,4]
    n_paras = 2 + p + q + (m+1)

    # Define CARMA GP likelihood function
    def fcoeff_ll(log_coeffs, y, gp):
        gp.kernel.set_log_fcoeffs(log_coeffs)
        ll = gp.log_likelihood(y, quiet = True)
        return ll


    # Initialize GP Kernel
    init_log_fcoeff = carma_log_fcoeff_init(p, q)
    init_log_ar, init_log_ma = CARMA_term.fcoeffs2carma_log(init_log_fcoeff, p)
    gp_kernel = CARMA_term(init_log_ar, init_log_ma)
    carma_gp = GP(gp_kernel, mean=0)

    # Define prior for MultiNest

    def myprior(cube):
        params = cube.copy()

        params[:p] = [cube[i]* 14 - 7 for i in range(p)]

        params[p: p+q+1] = [cube[i]* 10 - 5 for i in range(p,p+q+1)]

        #Micro-lensing coefficients
        params[p+q+1:p+q+1+m+1] = [cube[i] * 20 - 10 for i in range(p+q+1,p+q+1+m+1)]

        #Delta
        params[p+q+1+m+1] = cube[p+q+1+m+1] * 100

        return params

    def gp_timedelay(params,m=m,p=p,q=q, y=y, z=z, yerr=yerr, zerr=zerr, gp=carma_gp):

        log_carma_coeffs = np.array(params[:p+q+1])
        micro = params[p+q+1:p+q+1+m+1]
        Delta = params[p+q+1+m+1]

        mat = np.empty((len(t),m))
        for i in range(m):
            mat[:,i] = (t - Delta)**(i+1) - np.mean((t - Delta)**(i+1))

        #Perform QR parametrization of covariate matrix
        q_ = np.linalg.qr(mat)[0]

        y_centered = y
        z_centered = z - (micro[0] + np.dot(q_, micro[1:]))

        t_del = t - Delta
        idx = np.argsort(np.concatenate([t_del, t]))
        ordered_times = list(np.sort(np.concatenate([t_del, t])))
        yzcomb_centered = list(np.concatenate([z_centered,y_centered])[idx])
        err_comb = list(np.concatenate([zerr, yerr])[idx])

        yzcomb_centered = yzcomb_centered - np.median(yzcomb_centered)

        ll = -np.inf

        try:
            gp.compute(ordered_times, err_comb)
            ll = fcoeff_ll(log_carma_coeffs, yzcomb_centered, gp)
        except celerite.solver.LinAlgError as c:
            pass
        except Exception as e:
            pass

        return ll

    def parameter_names(m,p,q):
        #para_names  = ['Delta', 'mu', 'sigma', 'intercept']
        para_names = []
        for i in range(p+q+1):
            para_names.append("carma_"+str(i+1))
        para_names.append("theta0")
        for j in range(m):
            para_names.append("theta"+str(j+1))
        para_names.append("delta")
        return para_names

    #Initialize MultiNest
    parameters = parameter_names(m,p,q)
    n_params = len(parameters)
    # name of the output files]
    save_dir = './nest_output/'
    prefix = 'gp_CA_'+str(data_file[:-4])+'_c'+str(p)+str(q)+'m'+str(m)+'_'

    # Run MultiNest
    result = solve(LogLikelihood=gp_timedelay, Prior=myprior, n_live_points = 500, importance_nested_sampling = False, multimodal = True, 
        use_MPI = True, n_dims=n_params, verbose=False,outputfiles_basename=save_dir+prefix, resume=True)
    
    result_file_name = save_dir+prefix+'summary.txt'

    return process_gp_results.process_results_file(result_file_name, m, p ,q)

# %% Reformat function to allow vector input for parallelization
def fit_tdcarma_gp(vec):
    return fit_tdcarma_gp_(vec[0], vec[1], vec[2])

# %% Choose models to fit
pq = [[4,1]]
m = [3]

#%% Prepare combined input for simulation
cobi = list(itertools.product(pq, m))
cobi_vec = [[x[0][0], x[0][1], x[1]] for x in cobi]

# %% choose number of cores
n_cores = 6

# %% Fit model to data
if __name__ == '__main__':
    with Pool(n_cores) as p:
        #result = p.map(integ_lik, cobi)
        result = list(tqdm.tqdm(p.imap(fit_tdcarma_gp, cobi_vec), total = len(cobi_vec)))

# %% Final dataframe
final_df = pd.concat(result, ignore_index=True)

# %% pickle output
#save_dir = './results/'
save_dir = ''
filename = save_dir+data_file[:-4]+'_results.pkl'
final_df.to_pickle(filename)