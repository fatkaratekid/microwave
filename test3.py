import numpy as np
import matplotlib.pyplot as plt

# points = np.array([(1, 1), (2, 4), (3, 1), (9, 3)])
# # get x and y vectors
# x = points[:,0]
# y = points[:,1]
#
# # calculate polynomial
# z = np.polyfit(x, y, 3)
# f = np.poly1d(z)
#
# print type(z)
# print z

from pydub import AudioSegment
song = AudioSegment.from_mp3("audio.mp3")


from numpy import genfromtxt
data = genfromtxt('audiodata.csv', delimiter=',')

data = song[:10000]
# from scipy.io import wavfile
# fs, nin_data = wavfile.read("audio.mp3")

# data = np.random.uniform(-1,1,44100) # 44100 random samples between -1 and 1
# scaled = np.int16(data/np.max(np.abs(data)) * 32767)
from scipy.io.wavfile import write

print type(data)
total_points = data.shape[0]

sample_size = 35000
slice_size = 20
degrees = 10
multiplier = 1

x = np.linspace(1, slice_size, num=slice_size, endpoint=True)
#print "---------------"
#print single_slice
#print "vs"

t = np.reshape(data[0:sample_size],(int(sample_size/slice_size),slice_size))
i = 0
y_new = np.array([])
print type(y_new)
print y_new.shape


for row in t:
    i += 1
    z = np.polyfit(x, row, degrees)
#print z
#print "---------------"

    f = np.poly1d(z)

# calculate new x's and y's
    x_new = np.linspace(1, slice_size, num=slice_size * multiplier, endpoint=True)
    y_new = np.append(y_new, f(x_new))

x_new = np.linspace(1, sample_size, num=sample_size*multiplier, endpoint=True)
x = np.linspace(1, sample_size, num=sample_size, endpoint=True)
print x_new.shape
print y_new.shape

print (data[0:sample_size]).shape
print x.shape

# plt.plot(x,data[0:sample_size],'o', x_new, y_new)
# plt.xlim([x[0]-1, x[-1] + 1 ])
# plt.show()

write('test.wav', 8192 * multiplier, y_new)
