import numpy as np
import matplotlib.pyplot as plt

#CODE IS FROM AI (I am just using it to learn how the question is too be solved)

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

# ---- THIS PART WAS MISSING: build A, solve for theta, unpack m_hat/b_hat ----
A = np.column_stack([x, np.ones_like(x)])
theta, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
m_hat, b_hat = theta
# -------------------------------------------------------------------------

xs = np.linspace(x.min(), x.max(), 200)   # a fine grid of x-values to draw a smooth line
ys_fit = m_hat * xs + b_hat               # predicted y at each of those x-values

plt.scatter(x, y, label="Data")
plt.plot(xs, ys_fit, color="red", label=f"Fit: y = {m_hat:.1f}x + {b_hat:.0f}")
plt.xlabel("Square footage (sq ft)")
plt.ylabel("Sale price (USD)")
plt.legend()
plt.show()
