# %% [markdown]
# # Load Packages

# %%
from uacqr import uacqr
from helper import generate_data
from experiment import experiment

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial

# %% [markdown]
# # Define Data Generating Process

# %% [markdown]
# In a conditional gaussian setting, set the conditional expectation and the conditional noise

# %%
def cond_exp(x):
    return np.sin(1/(x[:,0]**3))

def noise_sd_fn(x):
    return 1*x[:,0]**2

# %%
x_dist = partial(np.random.beta, a=1.2, b=0.8)

# %%
n=100
T=800 # number of test points
p=1
n0 = int(n/2) # number of training points

# %% [markdown]
# # Simulate Data

# %%
np.random.seed(1)
data = generate_data(n+T, p, cond_exp, noise_sd_fn, x_dist)

# %%
x = data[0]
y = data[1]

if len(x.shape)==1:
    x = x.reshape(-1,1)

x_train = x[:n0]
y_train = y[:n0]

x_calib = x[n0:n]
y_calib = y[n0:n]

x_test = x[n:]
y_test = y[n:]

# %%
plt.scatter(x_train[:,0], y_train)
plt.xlabel('$X_{\cdot,1}$')
plt.ylabel('Y')
plt.title('Y versus X')

# %% [markdown]
# # Run LACQR

# %% [markdown]
# Define model hyperparameters

# %%
nn_params = {'dropout':0, 'epochs':1000, 'hidden_size':100, 'lr':1e-3, 'batch_norm':False, 
             'batch_size':2, 'normalize':True, 'weight_decay':0, 'epoch_model_tracking':True,
             "use_gpu":False}


# %% [markdown]
# Initialize uacqr class. We usually set $B$ to be one less than the number of epochs

# %%
uacqr_results = uacqr(nn_params,
                     model_type='neural_net',B=999, random_state=100, uacqrs_agg='std')

# %% [markdown]
# Fit the base quantile regressors

# %%
uacqr_results.fit(x_train, y_train)

# %% [markdown]
# Conformalize / calibrate the quantile regressors

# %%
uacqr_results.calibrate(x_calib, y_calib)

# %% [markdown]
# # Evaluation

# %% [markdown]
# Evaluate the various conformal methods on test data

# %%
print(uacqr_results.evaluate(x_test, y_test))

# %% [markdown]
# # Plot Results (for simulated data)

# %% [markdown]
# All in one chart

# %%
uacqr_results.plot(cond_exp, noise_sd_fn)

# %% [markdown]
# Graphic with simpler formatting

# %%
uacqr_results.plot_simple(cond_exp, noise_sd_fn,
                         xlabel='$X_{\cdot,1}$', expanded=True)
plt.tight_layout()


