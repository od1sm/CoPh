import numpy as np  # calculations
import pathlib  # needed to create folder
import matplotlib.pyplot as plt  # needed for graphs
import sympy as sp
from sympy.abc import E, e, n, M, o

print('Program started running!')

dpisiz = 1000


def savim(dir, name):
    path = pathlib.Path(f"./{dir}")
    path.mkdir(exist_ok=True, parents=True)
    plt.savefig(f'./{dir}/{name}.png', dpi=dpisiz)
    print(f'Saved image location: ./{dir}/{name}.png')


fract = e ** n / sp.factorial(n)
func = (sp.sin(M)) ** n
dif_func = (sp.Derivative(func, (M, n - 1)))
sumo = sp.Sum(fract * dif_func, (n, 1, o))
sum3 = (sumo.subs(o, 3)).doit()
sum10 = (sumo.subs(o, 10)).doit()
E3 = sum3 + M
E10 = sum10 + M
En3_e3 = sp.trigsimp(E3.subs(e, 0.3))
En10_e3 = sp.trigsimp(E10.subs(e, 0.3))
En3_e9 = sp.trigsimp(E3.subs(e, 0.9))
En10_e9 = sp.trigsimp(E10.subs(e, 0.9))

# Use sp.lambdify to create our lambda functions
# and plot our graphs. Could have used sp.plot but I prefer having more options with matplotlib.

numberoftries = 10000
xx = np.linspace(0, 2 * np.pi, 10000)
En3_e3f = sp.lambdify(M, En3_e3)(xx)
En10_e3f = sp.lambdify(M, En10_e3)(xx)
En3_e9f = sp.lambdify(M, En3_e9)(xx)
En10_e9f = sp.lambdify(M, En10_e9)(xx)

fig = plt.figure()
ax = plt.axes()

# Import data from task 1 to compare

data = np.genfromtxt(fname="data.csv", delimiter=',')
ax.set_ylabel('E')
ax.set_xlabel('M')
ax.plot(data[:, 0], data[:, 2], label='E for e=0.3', color='tab:pink')
ax.plot(xx, En3_e3f, label='E for n=3, e=0.3', linestyle='dashed', color='tab:red')
ax.plot(xx, En10_e3f, label='E for n=10, e=0.3', linestyle='dotted', color='tab:blue')
ax.legend()
savim('pr1_task2', 'e_03')
fig1 = plt.figure()
ax1 = plt.axes()
ax1.plot(data[:, 0], data[:, 5], label='E for e=0.9', color='black')
ax1.plot(xx, En3_e9f, label='E for n=3, e=0.9', color='tan')
ax1.plot(xx, En10_e9f, label='E for n=10, e=0.9', color='fuchsia')
ax1.legend()
ax1.set_ylabel('E')
ax1.set_xlabel('M')
savim('pr1_task2', 'e_09')

print('Program Finished running!')
# As we can see from our results, for lower eccentricities both n=3 and n=10 are nearly identical
# compared to the Newton-Raphson method. However for 0.9, this is no longer the case.
# In this case there are some sinusoidal curves with small amplitude appearing except the
# middle of the curve (There are still some curvature showing at the middle,
# however it’s close to the Newton-Raphson method).
