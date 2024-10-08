#code from https://www.youtube.com/watch?v=N7Oh0mk4YGc
# Python program to implement Runge Kutta method
# A sample differential equation "dy / dx = (x - y)/2"


import numpy as np
import matplotlib.pyplot as plt

def dydx(t, y):
    return y*t + y

def solveIVP(dydx, tspan, y0, h, solver):
    # Initialise t and y arrays
    t = np.arange(tspan[0], tspan[1] + h, h)
    y = np.zeros(len(t))
    y[0] = y0

    # Loop through steps and calculate single step solver solution
    for n in range(len(t) - 1):
        y[n + 1] = solver(dydx, t[n], y[n], h)

    return t, y

def rungeKutta( dydx, x0, y0, h ):


    k1 = h * dydx(x0, y0)
    k2 = h * dydx(x0 + 0.5 * h, y0 + 0.5 * k1)
    k3 = h * dydx(x0 + 0.5 * h, y0 + 0.5 * k2)
    k4 = h * dydx(x0 + h, y0 + k3)

    return y0 + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)




y0 = 1
h = 0.2
tspan = [0,2]


t, y = solveIVP(dydx, tspan, y0, h, rungeKutta)

#print table of solutions
print(" t | y/n------------------------------------ )")
for n in range(len(t)):
    print(f"{t[n]:0.1f} | {y[n]:10.6f}")


REAL_SOLUTION = pow(2.72,pow(t,2)/2+t)

x = np.linspace(0, 2, 100)



fig, (ax1, ax2) = plt.subplots(2, 1)


ax1.plot(t,y,"b-o")
ax1.set_title('Runge Kutta')
ax2.plot(t,REAL_SOLUTION )
ax2.set_title('real solution')

plt.xlabel("$t$")
plt.ylabel("$y$")

plt.tight_layout()
plt.show()


# This code is contributed by Prateek Bhindwar this coment is from geeks for geeks i have edited it some to make it work with the other code i have
#geeks for geeks