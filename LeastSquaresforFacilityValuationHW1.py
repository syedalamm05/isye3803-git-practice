import numpy as np
rng = np.random.default_rng(7) # keep fixed for reproducibility
N = 30
x = rng.uniform(800.0, 3200.0, size=N) # square feet
x = np.sort(x)
# "True" relationship used to generate synthetic observations
m_true = 310.0 # USD per sq ft (industrial/light commercial ballpark)
b_true = 60000.0 # USD (fixed component / site baseline)
# Heteroskedastic noise: percent-of-price noise (more realistic)
base_price = m_true * x + b_true
noise_frac = 0.06 # ~6% typical valuation/sale variability
y = base_price + rng.normal(0.0, noise_frac * base_price, size=N)
