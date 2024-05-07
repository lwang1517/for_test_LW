# %% Import packages
import matplotlib.pyplot as plt
import numpy as np
arr = np.array(("my favorite language is Python", "Python"))
arr

# %%
print("Hellow World")

# %%

def add_or_subtract(n1, n2, subtract=False):
    if subtract:
        return n1 - n2
    else:
        return n1 + n2


results = add_or_subtract(1, 2, True)
print(results)

results = add_or_subtract(n2=1, n1=2, subtract=True)
print(results)

# %%

xpts = np.array([1, 3])
ypts = np.array([2, 9])
plt.plot(xpts, ypts)
# %%
