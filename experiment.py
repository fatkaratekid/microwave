# import numpy as np
# from scipy.optimize import curve_fit
# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c
# xdata = np.linspace(0, 4, 50)
# y = func(xdata, 2.5, 1.3, 0.5)
# ydata = y + 0.2 * np.random.normal(size=len(xdata))
# popt, pcov = curve_fit(func, xdata, ydata)

import numpy as np
from scipy.interpolate import interp1d

x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x ** 2 / 9.0)
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')

xnew = np.linspace(0, 10, num=41, endpoint=True)

import matplotlib.pyplot as plt

plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()
