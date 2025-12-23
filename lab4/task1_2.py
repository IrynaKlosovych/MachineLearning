import numpy as np
import matplotlib.pyplot as plt

x = np.array([28, 14, 54, 16, 22, 15])
y = np.array([-15, 10, 4, 5, 11, 28])

a, b = np.polyfit(x, y, 1)

print(f"a = {a:.3f}, b = {b:.3f}")

x_line = np.linspace(min(x), max(x), 100)
y_line = a * x_line + b

plt.scatter(x, y, color='red', label='Експериментальні точки')
plt.plot(x_line, y_line, label='Апроксимація (МНК)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()