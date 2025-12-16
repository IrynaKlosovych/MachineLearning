import numpy as np
import matplotlib.pyplot as plt

def activation(value):
    return 1 if value >= 0 else 0

def or_func(x1, x2, w1=1, w2=1, b=-0.5):
    s = x1 * w1 + x2 * w2 + b
    return activation(s)

def and_func(x1, x2, w1=1, w2=1, b=-1.5):
    s = x1 * w1 + x2 * w2 + b
    return activation(s)
    
def not_func(x,  w = -1, b = 0.5): 
    s = x * w + b 
    return activation(s)

def xor_func(x1, x2):
    or_res = or_func(x1, x2)
    and_res = and_func(x1, x2)
    nand_res = not_func(and_res)
    return and_func(or_res, nand_res)
    
def plot_perceptron_with_bg(ax, func, lines, title, resolution=100):
    x_vals = np.linspace(0, 1, resolution)
    y_vals = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(x_vals, y_vals)
    zz = np.zeros_like(xx)

    for i in range(resolution):
        for j in range(resolution):
            zz[i,j] = func(xx[i,j], yy[i,j])

    ax.contourf(xx, yy, zz, alpha=0.3, levels=[-0.1,0.5,1.1], colors=["red","blue"])

    inputs = [(0,0), (0,1), (1,0), (1,1)]
    class0 = [x for x in inputs if func(*x) == 0]
    class1 = [x for x in inputs if func(*x) == 1]
    class0 = np.array(class0)
    class1 = np.array(class1)
    ax.scatter(class0[:,0], class0[:,1], label="class 0", color="red", s=100)
    ax.scatter(class1[:,0], class1[:,1], label="class 1", color="blue", s=100)

    random_points = np.random.rand(50, 2)
    random_labels = np.array([func(x,y) for x,y in random_points])
    ax.scatter(random_points[random_labels==0,0], random_points[random_labels==0,1], color="red", alpha=0.5)
    ax.scatter(random_points[random_labels==1,0], random_points[random_labels==1,1], color="blue", alpha=0.5)

    x_line = np.linspace(-0.2, 1.2, 100)
    if lines is not None:
        for line in lines:
            w = line["w"]
            b = line["b"]
            label = line.get("label", "decision boundary")
            if w[1] != 0:
                y_line = (-w[0]*x_line - b)/w[1]
                ax.plot(x_line, y_line, label=label)
            else:
                x_vert = -b / w[0]
                ax.axvline(x=x_vert, label=label, color="green")

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.legend()
    ax.grid()

lines_or = [{"w": [1, 1], "b": -0.5, "label": "OR boundary"}]
lines_and = [{"w": [1, 1], "b": -1.5, "label": "AND boundary"}]
lines_xor = [
    {"w": [1, 1], "b": -0.5, "label": "OR boundary"},
    {"w": [1, 1], "b": -1.5, "label": "AND boundary"}
]

fig, axs = plt.subplots(3, 1, figsize=(8, 16))

plot_perceptron_with_bg(axs[0], or_func, lines_or, "Perceptron OR")
plot_perceptron_with_bg(axs[1], and_func, lines_and, "Perceptron AND")
plot_perceptron_with_bg(axs[2], xor_func, lines_xor, "Perceptron XOR")

plt.tight_layout()
plt.show()
