
import numpy as np
import matplotlib.pyplot as plt

history = np.load('hist.npy')

epochs = range(150)

plt.plot(epochs,history,'b')

plt.xlabel('Epoch')
plt.ylabel('Points Scored')

plt.savefig('longer_performance_plot.png')

plt.show()