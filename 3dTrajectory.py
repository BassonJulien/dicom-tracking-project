import matplotlib
import notebook as notebook
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
# zline = np.linspace(0, 15, 1000)
# xline = np.sin(zline)
# yline = np.cos(zline)

xline = [0, 1, 0]

yline = [2, 2, 3]

zline = [1, 2, 2]

ax.plot3D(xline, yline, zline, 'black')


plt.show(fig)