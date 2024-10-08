#code from https://www.youtube.com/watch?v=N7Oh0mk4YGc

import numpy as np
import matplotlib.pyplot as plt
import math as mth


def solveIVP(f, tspan, y0, h, solver):
    # Initialise t and y arrays
    t = np.arange(tspan[0], tspan[1] + h, h)
    y = np.zeros(len(t))
    y[0] = y0

    # Loop through steps and calculate single step solver solution
    for n in range(len(t) - 1):
        y[n + 1] = solver(f, t[n], y[n], h)

    return t, y

def euler(f, t, y, h):
    return y + h * f(t, y)

def f(t,y):
    return mth.cos(pow(y,2))*t + y

tspan = [0,5]
y0 = 1
h = 0.2

#solve Ivp
t, y = solveIVP(f, tspan, y0, h, euler)

#print table of solutions
print(" t | y/n------------------------------------ )")
for n in range(len(t)):
    print(f"{t[n]:0.1f} | {y[n]:10.6f}")

fig, ax = plt.subplots()
plt.plot(t,y,"b-o")
plt.xlabel("$t$")
plt.ylabel("$y$")
plt.show()