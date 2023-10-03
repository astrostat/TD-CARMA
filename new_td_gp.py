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
#from mpi4py import MPI
plt.style.use('tableau-colorblind10')

import eztao
import celerite
from celerite import GP
from eztao.ts import carma_log_fcoeff_init
from eztao.carma import DRW_term,DHO_term, CARMA_term
from eztao.ts import neg_fcoeff_ll
from oldscripts import helper_functions

def fcoeff_ll(log_coeffs, y, gp):
    gp.kernel.set_log_fcoeffs(log_coeffs)
    ll = gp.log_likelihood(y, quiet = True)
    return ll



#%% Choose and import data
data_file = '2M1134_WFI.csv'
data = pd.read_csv('./td_carma/data/2M1134_WFI.csv')

#%% Extract data
t = data.iloc[:,0] - min(data.iloc[:,0])
y = data.iloc[:,1]
z = data.iloc[:,3]
yerr = data.iloc[:,2]
zerr = data.iloc[:,4]

#%% Choose (p,q,m) triplet
p = 2
q = 1
m = 3

n_paras = 2 + p + q + (m+1)

#%% Initialize GP and log-prob function according to p and q
init_log_fcoeff = carma_log_fcoeff_init(p, q)
init_log_ar, init_log_ma = CARMA_term.fcoeffs2carma_log(init_log_fcoeff, p)
if p == 1:
    gp_kernel = DRW_term(init_log_ar, init_log_ma)
if p == 2:
    gp_kernel = DHO_term(init_log_ar[0], init_log_ar[1], init_log_ma[0], init_log_ma[1])
else:
    gp_kernel = CARMA_term(init_log_ar, init_log_ma)
carma_gp = GP(gp_kernel, mean = np.median(y))

def fcoeff_ll(log_coeffs, y, gp, p):
    if p <= 2:
        gp.set_parameter_vector(log_coeffs)
    else:
        gp.kernel.set_log_fcoeffs(log_coeffs)
    ll = gp.log_likelihood(y, quiet = True)
    return ll


#%% Write prior and likelihood function for normal (unknown mean, known variance)
def myprior(cube):
    params = cube.copy()

    #CARMA log-coefficients (polynomial form)
    #params[:p+q+1] = [cube[i]* 16 - 8 for i in range(p+q+1)]

    params[:p] = [cube[i]* 14 - 7 for i in range(p)]

    params[p: p+q+1] = [cube[i]* 10 - 5 for i in range(p,p+q+1)]

    #Micro-lensing coefficients
    params[p+q+1:p+q+1+m+1] = [cube[i] * 100 - 50 for i in range(p+q+1,p+q+1+m+1)]

    #Delta
    params[p+q+1+m+1] = cube[p+q+1+m+1] *  100

    return params

#%% Time Delay likelihood
def gp_timedelay(params,m,p,q, y, z, yerr, zerr, gp):

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

    # #Compute gp
    # gp.compute(ordered_times, err_comb)

    # #Compute negative log-likelihood
    # ll = fcoeff_ll(log_carma_coeffs, yzcomb_centered, gp)

    ll = -1e30

    try:
        gp.compute(ordered_times, err_comb)
        ll = fcoeff_ll(log_carma_coeffs, yzcomb_centered, gp, p)
    except celerite.solver.LinAlgError as c:
        # print(c)
        pass
    # except Exception as e:
    #     pass

    return ll



def wrapped_gp_loglike(params):
    return gp_timedelay(params,m,p,q, y, z, yerr, zerr, carma_gp)

#%% Function to create array of names for parameters
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

#%% Initialize MultiNest
parameters = parameter_names(m,p,q)
n_params = len(parameters)
# name of the output files]
save_dir = './nest_output/'
#data_file = 'test81818181818'
prefix = 'gp_BA_'+str(data_file[:-4])+'_c'+str(p)+str(q)+'m'+str(m)+'_'


start = time.time()
#%%run MultiNest
result = solve(LogLikelihood=wrapped_gp_loglike, Prior=myprior, n_live_points = 1000, importance_nested_sampling = False, multimodal = True, 
    use_MPI = True, n_dims=n_params, verbose=True,outputfiles_basename=save_dir+prefix, resume=False)


stop = time.time()
print('Elapsed :', (stop-start)/60, ' min')

#%%
print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')
for name, col in zip(parameters, result['samples'].transpose()):
	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
# %%
