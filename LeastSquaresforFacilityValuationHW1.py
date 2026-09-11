import numpy as np
import matplotlib.pyplot as plt

#CODE IS FROM AI I AM JUST USING IT TO LEARN THE QUESTION 
# ---------------------------------------------------------------------------
# Data generation (provided)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)  # keep fixed for reproducibility
N = 30
x = rng.uniform(800.0, 3200.0, size=N)  # square feet
x = np.sort(x)

# "True" relationship used to generate synthetic observations
m_true = 310.0    # USD per sq ft (industrial/light commercial ballpark)
b_true = 60000.0  # USD (fixed component / site baseline)

# Heteroskedastic noise: percent-of-price noise (more realistic)
base_price = m_true * x + b_true
noise_frac = 0.06  # ~6% typical valuation/sale variability
y = base_price + rng.normal(0.0, noise_frac * base_price, size=N)

# ---------------------------------------------------------------------------
# Question 2/3: Matrix form + solve with lstsq (clean data)
# ---------------------------------------------------------------------------
A = np.column_stack([x, np.ones_like(x)])          # A has columns [x_i, 1]
theta, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
m_hat, b_hat = theta

print("Clean fit: m_hat = %.3f, b_hat = %.3f" % (m_hat, b_hat))

xs = np.linspace(x.min(), x.max(), 200)  # fine grid of x-values for plotting
ys_fit = m_hat * xs + b_hat              # predicted y at each of those x-values

plt.figure(figsize=(7, 5))
plt.scatter(x, y, label="Data")
plt.plot(xs, ys_fit, color="red", label=f"LS fit: y = {m_hat:.1f}x + {b_hat:.0f}")
plt.xlabel("Square footage (sq ft)")
plt.ylabel("Sale price (USD)")
plt.title("Least Squares Fit (clean data)")
plt.legend()
plt.tight_layout()
plt.savefig("fit_clean.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# Question 4: Introduce a single outlier and re-solve
# ---------------------------------------------------------------------------
y_outlier = y.copy()          # keep the original y untouched
outlier_idx = 15              # pick one property to corrupt (index 0-29)
y_outlier[outlier_idx] = y[outlier_idx] * 100

A2 = np.column_stack([x, np.ones_like(x)])   # x unchanged, so A2 == A
theta2, residuals2, rank2, sv2 = np.linalg.lstsq(A2, y_outlier, rcond=None)
m_hat2, b_hat2 = theta2

print("Outlier fit: m_hat2 = %.3f, b_hat2 = %.3f" % (m_hat2, b_hat2))
print("Corrupted point -> x0 = %.1f, y0_original = %.1f, y0_corrupted = %.1f"
      % (x[outlier_idx], y[outlier_idx], y_outlier[outlier_idx]))

ys_fit2 = m_hat2 * xs + b_hat2

# Full-scale view: shows how extreme the outlier is relative to everything else
plt.figure(figsize=(7, 5))
plt.scatter(x, y_outlier, label="Data (with outlier)")
plt.scatter([x[outlier_idx]], [y_outlier[outlier_idx]], color="orange",
            zorder=5, label="Outlier point")
plt.plot(xs, ys_fit, color="red", linestyle="--",
         label=f"Original fit: y={m_hat:.1f}x+{b_hat:.0f}")
plt.plot(xs, ys_fit2, color="green",
         label=f"Fit with outlier: y={m_hat2:.1f}x+{b_hat2:.0f}")
plt.xlabel("Square footage (sq ft)")
plt.ylabel("Sale price (USD)")
plt.title("Effect of a Single Outlier on Least Squares Fit")
plt.legend()
plt.tight_layout()
plt.savefig("fit_outlier.png", dpi=150)
plt.show()

# Zoomed view: shows original data range so you can see how far the new
# line has shifted away from the "good" points
plt.figure(figsize=(7, 5))
plt.scatter(x, y, label="Data (original, uncorrupted)")
plt.plot(xs, ys_fit, color="red", linestyle="--",
         label=f"Original fit: y={m_hat:.1f}x+{b_hat:.0f}")
plt.plot(xs, ys_fit2, color="green",
         label=f"Fit with outlier: y={m_hat2:.1f}x+{b_hat2:.0f}")
plt.xlabel("Square footage (sq ft)")
plt.ylabel("Sale price (USD)")
plt.title("Effect of Outlier on Fitted Line (zoomed to original data range)")
plt.legend()
plt.tight_layout()
plt.savefig("fit_outlier_zoom.png", dpi=150)
plt.show()
