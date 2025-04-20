import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Test Plot')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)

# Save the plot
plt.savefig('test_plot.png')
print("Plot saved as test_plot.png") 