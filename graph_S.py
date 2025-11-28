import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('S0_vs_q.txt')
x = data[:,0]
y = data[:,1] - 1


plt.scatter(x,y,label='Structure factor')

plt.plot(x,y,label='Structure factor')

plt.xlabel('Wave number')
plt.ylabel('S(q,0)')
plt.savefig('S_q.png')
plt.show()
