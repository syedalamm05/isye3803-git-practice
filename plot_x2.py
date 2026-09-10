import matplotlib.pyplot as plt
import numpy as np

# Create x values from -10 to 10
x = np.linspace(-10, 10, 100)

# Calculate y values using f(x) = x^2
y = x**2

# Create the plot
plt.plot(x, y)

# Add labels and title
plt.xlabel('x')
plt.ylabel('f(x) = x²')
plt.title('Graph of f(x) = x²')

# Display the plot
plt.show()
