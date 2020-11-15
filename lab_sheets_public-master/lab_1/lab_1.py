# %% Imports
import numpy as np
from scipy import stats
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from pprint import pprint

### %% Matrix operations ###

A = np.matrix('2 3; 4 -1; 5 6')
B = np.matrix([[5, 2], [8, 9], [2, 1]])

print(A)
print(B)
print(A.shape)
print(B.shape)

C = 3 * A
print(C)
C = A + B
print(C)
C = A * B.transpose()
print(C)

pprint(A.sum())
pprint(A.mean())
pprint(A.var())

#############################


### %%Loading Data ###


D = np.loadtxt('data.dat',delimiter=',')
D.shape

#####################




### Scatter Plots ###

# %% 2D
x = D[:, 0]
y = D[:, 1]
fig, ax = plt.subplots()
ax.scatter(x,y)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xbound(0,25)
ax.set_ybound(-30,40)
ax.grid(True, color='g')
plt.show()

# %% 3D
x = D[:, 3]
y = D[:, 2]
z = D[:, 1]
fig, ax = plt.subplots(subplot_kw={'projection' : '3d'})
ax.scatter(x,y,z)
plt.show()

####################


### %% Histograms ###

X = D[:, 0]
fig, ax = plt.subplots()
plt.hist(X)

##################



### %% Normal Distributions ###

ND = np.random.randn(1000)

fig, ax = plt.subplots()
ax.set_xbound(-5,5)
plt.hist(ND)

##################



### %% Saving Outputs ###

np.savetxt('normaldist.DAT', ND, delimiter=',')

######################



### %% Random Data ###

UD = np.random.rand(100)

fig, ax = plt.subplots()
plt.hist(UD)

######################
