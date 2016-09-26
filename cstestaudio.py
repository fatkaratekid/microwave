import numpy as np
from scipy.io.wavfile import write
from numpy import genfromtxt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import inspect
from scipy.interpolate import CubicSpline



data = genfromtxt('audiodata.csv', delimiter=',');

# data = np.random.uniform(-1,1,44100) # 44100 random samples between -1 and 1
# scaled = np.int16(data/np.max(np.abs(data)) * 32767)
write('test.wav', 8192, data)

print type(data)
#total_points = data.shape[0]

single_slice = data[:10000]
slice_total_points = single_slice.shape[0]

print "total slice points: " + str(slice_total_points)
x = np.linspace(1, slice_total_points, num=slice_total_points, endpoint=True)

#print x
#print (np.cos(-x ** 2 / 9.0)).shape

cs = CubicSpline(x, single_slice)

#f = interp1d(x, single_slice)
#f2 = interp1d(x, single_slice, kind='cubic')

#print f
#inspect.getsourcelines(cs)

xnew = np.linspace(1, slice_total_points, num=slice_total_points*4, endpoint=True)


#plt.plot(x, single_slice, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
#plt.legend(['data', 'linear', 'cubic'], loc='best')
#plt.show()
