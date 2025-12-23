import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y = np.array([3.2, 3, 1, 1.8, 1.9])

X = np.vander(x, len(x))

coeffs = np.linalg.solve(X, y)

print("Коефіцієнти полінома:")
print(coeffs)

def P(x):
    return np.polyval(coeffs, x)

x_test = np.array([0.2, 0.5])
y_test = P(x_test)

print("\nЗначення в точках:")
for xi, yi in zip(x_test, y_test):
    print(f"P({xi}) = {yi}")

x_plot = np.linspace(min(x), max(x), 400)
y_plot = P(x_plot)

plt.scatter(x, y, color='red', label='Табличні точки')
plt.plot(x_plot, y_plot, label='Інтерполяційний поліном 4-го степеня для 5 точок')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

