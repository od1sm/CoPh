import pathlib  # needed to create folder
import matplotlib.pyplot as plt  # needed for graphs
from tqdm import tqdm
import numpy as np

print('Program started running!')

pi = np.pi
dpisiz = 1000  # for images resolution

#######################################
# Lambda anonymous functions          #
#######################################

f = lambda x, e, k: x - k - e * np.sin(x)
df = lambda x, e: 1 - e * np.cos(x)


########################################
# Creating Our Newton-Raphson function #
########################################
def newton_raphson(f, df, x, e, TOL, k):
    error = 1
    iterations = 0
    while error > TOL:
        new_x = x - f(x, e, k) / df(x, e)
        error = abs(new_x - x)
        x = new_x
        iterations += 1
        # print(f'x{iterations}: {x:.15f} \t | \t error: {error} \t | \t f(x):{f(x, e,k)} \t | \t df(x):{df(x, e)} \t | \t line:{df(x, e)} * x+{f(x, e,k)}')
    # print(f"Newton's Estimate = {x:.15f}\nIterations: {iterations}")
    return x


#####################################
# Function to save images to folder #
#####################################
def savim(dir, name):
    path = pathlib.Path(f"./{dir}")
    path.mkdir(exist_ok=True,  # Without exist_ok=True,FileExistsError show up if folder already exists
               parents=True)  # Missing parents of the path are created.
    plt.savefig(f'./{dir}/{name}.png', dpi=dpisiz)
    print(f'Saved image at location: /{dir}/{name}.png')


numberoftries = 10000
i = np.empty((numberoftries, 6))
i[:, 0] = np.linspace(0, 2 * pi, numberoftries)
e = [0.1, 0.3, 0.5, 0.7, 0.9]

for j in tqdm(range(1, 6)):
    for k in range(numberoftries):
        i[k, j] = newton_raphson(f, df, 1, e[j - 1], 1e-15, i[k, 0])  # Adding NP-estimates to array
e = [0.1, 0.3, 0.5, 0.7, 0.9]

# Plotting our function
fig = plt.figure()
ax = plt.axes()
lines = ax.plot(i[:, 0], i[:, 0:5])
ax.set_ylabel('E')
ax.set_xlabel('M')
ax.set_title('Project 1 - Task 1 - Newton-Raphson method')
labels = [0] * 5
for o in range(0, 5):
    labels[o] = f'E(M) for e={e[o]}'
ax.legend(lines, labels)
savim('./pr1_task1/images', 'plot')

print('Program finished running!')

##############################################################################
#    A faster convergence could be achieved with a better initial value.     #
#          Current code uses a standard initial value of one.                #
##############################################################################
