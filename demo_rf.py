# %% [markdown]
# # Load Packages

# %%
from uacqr import uacqr
from helper import generate_data

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial

# %% [markdown]
# # Define Data Generating Process (for simulated settings)

# %% [markdown]
# In a conditional gaussian setting, set the conditional expectation and the conditional noise

# %%
def cond_exp(x):
    return (x[:,0]>-0)*1 # Generally we only make function of first covariate
def noise_sd_fn(x):
    return 0.01 + np.sin(15*x[:,0])**2*(x[:,0]>0) # Like above, only function of first covariate 

# %%
x_dist = partial(np.random.uniform, low=-1, high=1)

# %%
n=200
T=200 # number of test points
p=100
n0 = int(n/2) # number of training points

# %% [markdown]
# # Simulate Data

# %%
np.random.seed(3)
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
# # Run UACQR

# %% [markdown]
# Define model hyperparameters

# %%
params_qforest = dict()
params_qforest["min_samples_leaf"] = 5
params_qforest["max_features"] = 1.

# %% [markdown]
# Initialize uacqr class.

# %%
uacqr_results = uacqr(params_qforest,
                     model_type='rfqr',B=100, random_state=0, uacqrs_agg='iqr')

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


