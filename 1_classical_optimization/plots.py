import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from problems import *

# how many steps and at what resolution do I want to plot
steps = 350
x = 0.1*np.arange(-steps/2, steps/2)
y = 0.1*np.arange(-steps/2, steps/2)
X, Y = np.meshgrid(x, y)

# defining the figure parameters
fig = plt.figure(figsize=(15, 20))

# first subplot of four
ax1 = fig.add_subplot(411, projection='3d')
surf1 = ax1.plot_surface(X, Y, problem_0(x, y),
                         cmap=cm.coolwarm,
                         linewidth=0,
                         antialiased=False)
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')
ax1.set_title('Problem 0 surface')
fig.colorbar(surf1, shrink=0.4, aspect=5, label='Z-Axis values')

# second subplot of four
ax2 = fig.add_subplot(412, projection='3d')
surf2 = ax2.plot_surface(X, Y, problem_1(x, y),
                         cmap=cm.coolwarm,
                         linewidth=0,
                         antialiased=False)
fig.colorbar(surf2,  shrink=0.4, aspect=5, label='Z-Axis values')
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_zlabel('Z axis')
ax2.set_title('Problem 1 surface')

# third subplot of four
ax3 = fig.add_subplot(413, projection='3d')
surf3 = ax3.plot_surface(X, Y, problem_2(x, y),
                         cmap=cm.coolwarm,
                         linewidth=0,
                         antialiased=False)
fig.colorbar(surf3, shrink=0.4, aspect=5, label='Z-Axis values')
ax3.set_xlabel('X axis')
ax3.set_ylabel('Y axis')
ax3.set_zlabel('Z axis')
ax3.set_title('Problem 2 surface')

# fourth subplot of four
ax4 = fig.add_subplot(414, projection='3d')
surf4 = ax4.plot_surface(X, Y, problem_3(x, y),
                         cmap=cm.coolwarm,
                         linewidth=0,
                         antialiased=False)
fig.colorbar(surf3, shrink=0.4, aspect=5, label='Z-Axis values')
ax4.set_xlabel('X axis')
ax4.set_ylabel('Y axis')
ax4.set_zlabel('Z axis')
ax4.set_title('Problem 3 surface')

# and show plots
plt.show()
