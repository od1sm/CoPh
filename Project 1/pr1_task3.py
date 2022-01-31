import sympy as sp  # needed for symbolic interpretation
import pathlib  # needed to create folder
import matplotlib.pyplot as plt  # needed for graphs
from sympy.abc import E, M, j, e, k, n, o, x, i, d
import numpy as np
import os
from tqdm import tqdm

print('Program started running!')
#######################
# Some starting setup #
#######################
dpisiz = 1000
# clrs is created to rotate through some colors I prefer when using sympy.plot and matplotlib.plot
clrs = [(1, 'tab:blue'), (2, 'tab:orange'), (3, 'tab:green'), (4, 'tab:red'), (5, 'tab:purple'), (6, 'tab:gray')]
# lnsstyle is needed to switch linestyle when using matplotlib
lnsstyle = [0] * 6
lnsstyle = ['-', '--', '-.', ':', '--', '-']


#####################################
# Function to save images to folder #
#####################################

def savim(dir, name):
    path = pathlib.Path(f"./{dir}")
    path.mkdir(exist_ok=True, parents=True)
    plt.savefig(f'./{dir}/{name}.png', dpi=dpisiz)
    print(f'Saved image location: ./{dir}/{name}.png')


###################################################################
#                           PART I                                #
###################################################################

###############################################################
# Splitting Sum to two products to make it cleaner to look at #
###############################################################

jn_frc1 = (-1) ** k / (sp.factorial(k) * sp.factorial(n + k))
jn_frc2 = sp.Pow(x / 2, (n + 2 * k))
sum = sp.Sum(jn_frc1 * jn_frc2, (k, 0, o))

bessel = [0] * 6  # Zero list to fill with the bessel functions
plots_bsl = [0] * 6  # Zero list to fill with the bessel plots
knumber = 50  # overflows at 83 firstly for J_5

#########################################################################
#          Using symply.plot to plot our Bessel functions               #
#########################################################################
for b in tqdm(range(6)):
    bessel[b] = sp.simplify(sum.subs([(o, knumber), (n, b)]).doit())
    plots_bsl[b] = sp.plot(bessel[b], (x, 0.001, 15), show=False, line_color=clrs[b][1], label=f'J_{b}(x)', legend=True)
    if b == 0:
        print("\n Our initial plot, so appending not possible")  # Appending other plots to the first one
    else:
        plots_bsl[0].append(plots_bsl[b][0])

#########################################################################
#          Setting ylabel, saving and showing our plot                  #
#########################################################################
plots_bsl[0].ylabel = 'J_n(x)'
savim('pr1_task3', f'img_{knumber}')
# plots_bsl[0].save(f'pr1_task3/img_{knumber}')
# plots_bsl[0].show()

##################################################################################
# Because sympy.plot doesn't have some features I use (e.g. dpi, linestyle, etc) #
# I switch to matploblib (which is the backend of sympy.plot) by switching to    #
# the lambda (anonymous) function using sp.lambdify to a linspace of domain (0,15) #
##################################################################################

xx = np.linspace(0, 15, 10000)
lmbd_plots_bsl = [0] * 6  # Creating an empty array to fill with the lambda functions

print('Lambdifying our Sympy Sums')
for b in tqdm(range(0, 6)):
    lmbd_plots_bsl[b] = sp.lambdify(x, sp.simplify(sum.subs([(o, knumber), (n, b)]).doit()))(xx)

# Creating our figure
fig = plt.figure()
ax = plt.axes()

print('Plotting our bessel functions')
# Plotting our bessel functions
for b in tqdm(range(0, 6)):
    ax.plot(xx, lmbd_plots_bsl[b], color=clrs[b][1], label=f'J_{b}(x)', linestyle=lnsstyle[b])
ax.legend()
ax.set_title("Problem 1 - Task 3 - Bessel functions of the first kind")
ax.set_ylabel('J_n(x)')
ax.set_xlabel('x')
savim('pr1_task3', f'limg_{knumber}')

###################################################################
#                           PART II                               #
###################################################################

d = knumber  # kmax

# Interpreting J_j earlier for better readability
J_j = sp.Sum((-1) ** knumber / (sp.factorial(k) * sp.factorial(j + k)) * (j * e / 2) ** (j + 2 * k), (k, 0, d))

# Our sum with j our index of summation starting from 1 to o
sumE = M + sp.Sum(2 / j * J_j * sp.sin(j * M), (j, 1, o))
# Substituting the eccentricities e=0.3 and e=0.9
# and our upper bound of summation which is either 3 or 10
sum_j3_e03 = sp.simplify(sumE.subs([(e, 0.3), (o, 3)]).doit())
sum3_j10_e03 = sp.simplify(sumE.subs([(e, 0.3), (o, 10)]).doit())
sum_j3_e09 = sp.simplify(sumE.subs([(e, 0.9), (o, 3)]).doit())
sum3_j10_e09 = sp.simplify(sumE.subs([(e, 0.9), (o, 10)]).doit())

# Switching to lambda (anonymous) function
xxx = np.linspace(0, 2 * np.pi, 10000)

lmbd_sum_j3_e03 = sp.lambdify(M, sum_j3_e03)(xxx)
lmbd_sum3_j10_e03 = sp.lambdify(M, sum3_j10_e03)(xxx)
lmbd_sum_j3_e09 = sp.lambdify(M, sum_j3_e09)(xxx)
lmbd_sum3_j10_e09 = sp.lambdify(M, sum3_j10_e09)(xxx)

# Getting data from Task 1 to compare

data = np.genfromtxt(fname="data.csv", delimiter=',')
fig2 = plt.figure()
ax2 = plt.axes()
ax2.set_ylabel('E')
ax2.set_title("Problem 1 - Task 3 - Fourier-Bessel expansion of E with e=0.3")
ax2.set_xlabel('M')
ax2.plot(data[:, 0], data[:, 2], label='E for e=0.3', color='tab:pink')
ax2.plot(xxx, lmbd_sum_j3_e03, label='E for j=3, e=0.3', linestyle='dashed', color='tab:red')
ax2.plot(xxx, lmbd_sum3_j10_e03, label='E for j=10, e=0.3', linestyle='dotted', color='tab:blue')
ax2.legend()
savim('pr1_task3', 'e_03')
# fig2.show()
fig3 = plt.figure()
ax3 = plt.axes()
ax3.set_ylabel('E')
ax3.set_xlabel('M')
ax3.set_title("Problem 1 - Task 3 - Fourier-Bessel expansion of E with e=0.9")
ax3.plot(data[:, 0], data[:, 5], label='E for e=0.9', color='black')
ax3.plot(xxx, lmbd_sum_j3_e09, label='E for j=3, e=0.9', linestyle='dashed', color='tab:red')
ax3.plot(xxx, lmbd_sum3_j10_e09, label='E for j=10, e=0.9', linestyle='dotted', color='tab:blue')
ax3.legend()
savim('pr1_task3', 'e_09')
# fig3.show()

####################################################################################################
# As we can see from our results, for lower eccentricities both n=3 and n=10 are nearly            #
# identical compared to the Newton-Raphson method.                                                 #
# However for 0.9, this is no longer the case. In this case there are sinusoidal curves appearing  #
# everywhere in the curve with different amplitude (Higher at the ends).                            #
####################################################################################################


# Getting data from Task 2 to compare
# Starting with e=0.9

En3_e9f = np.genfromtxt(os.path.join(os.getcwd(), "pr1_task2data", "En3_e9f.csv"), delimiter=',')
En10_e9f = np.genfromtxt(os.path.join(os.getcwd(), "pr1_task2data", "En10_e9f.csv"), delimiter=',')
# Creating our figure
fig4 = plt.figure()
ax4 = plt.axes()
ax4.set_ylabel('E')
ax4.set_xlabel('M')
ax4.set_title("Problem 1 - Task 3 - Comparing with Task 2 for e=0.9")
ax4.plot(data[:, 0], En3_e9f[:], label='E(Task2) for n=3, e=0.9 ', color='black')
ax4.plot(data[:, 0], En10_e9f[:], label='E(Task2) for n=10, e=0.9', color='tab:pink')
ax4.plot(xxx, lmbd_sum_j3_e09, label='E for j=3, e=0.9', linestyle='dashed', color='tab:red')
ax4.plot(xxx, lmbd_sum3_j10_e09, label='E for j=10, e=0.9', linestyle='dotted', color='tab:blue')
ax4.legend()
savim('pr1_task3', 'e_09withtask2data')
# fig4.show()

# Doing e=0.3 now

En3_e3f = np.genfromtxt(os.path.join(os.getcwd(), "pr1_task2data", "En3_e3f.csv"), delimiter=',')
En10_e3f = np.genfromtxt(os.path.join(os.getcwd(), "pr1_task2data", "En10_e3f.csv"), delimiter=',')
# Creating our figure
fig5 = plt.figure()
ax5 = plt.axes()
ax5.set_ylabel('E')
ax5.set_xlabel('M')
ax5.set_title("Problem 1 - Task 3 - Comparing with Task 2 for e=0.3")
ax5.plot(data[:, 0], En3_e3f[:], label='E(Task2) for n=3, e=0.3 ', linestyle=':', color='black')
ax5.plot(data[:, 0], En10_e3f[:], label='E(Task2) for n=10, e=0.3', color='tab:pink')
ax5.plot(xxx, lmbd_sum_j3_e03, label='E for j=3, e=0.3', linestyle='dashed', color='tab:red')
ax5.plot(xxx, lmbd_sum3_j10_e03, label='E for j=10, e=0.3', linestyle='dotted', color='tab:blue')
ax5.legend()
savim('pr1_task3', 'e_03withtask2data')
# fig5.show()

print('Program finished running!')

#########################################################################################
# As we can see from our results, for lower eccentricities both n=3 and n=10 are nearly #
# identical compared to Kepler's equation which is created using Langrange's theorem.   #
#########################################################################################

##########################################################################################
# So as we can see our Fourier-Bessel expansion can be used for low values of e.         #
# With e increasing, this method doesn't perform as well when compared to Newton-Raphson #
# or when using Lagrange's Theorem                                                       #
##########################################################################################
