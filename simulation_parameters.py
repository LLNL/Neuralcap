import numpy as np

current_units_change = 2.0e-8
# current_units_change = 2.0e-4 # Worked for 50 and 500
n_steps = 1000
tlimit = 1.0
n_cycles = 4
# scan_rates = np.array([100, 200, 400, 600, 800, 1000])
# scan_rates = np.array([5, 10, 15, 20, 50, 100, 200, 500])
scan_rates = np.array([500])
N = 100
