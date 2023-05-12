import numpy as np
import matplotlib.pyplot as plt

# NOTE: these lines define global figure properties used for publication.
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg') # print figures in svg format
plt.rcParams.update({'font.size':14}) # set global font size



# the scalars
l1 = 1
l2 = 2
l3 = -3

# the vectors
v1 = np.array([4,5,1])
v2 = np.array([-4,0,-4])
v3 = np.array([1,3,2])

# linear weighted combination
l1*v1 + l2*v2 + l3*v3




# points (in Cartesian coordinates)
p = (3,1)
q = (-6,2)

plt.figure(figsize=(6,6))

# draw points
plt.plot(p[0],p[1],'ko',markerfacecolor='k',markersize=10,label='Point p')
plt.plot(q[0],q[1],'ks',markerfacecolor='k',markersize=10,label='Point q')

# draw basis vectors
plt.plot([0,0],[0,1],'k',linewidth=3,label='Basis S')
plt.plot([0,1],[0,0],'k',linewidth=3)

plt.plot([0,3],[0,1],'k--',linewidth=3,label='Basis T')
plt.plot([0,-3],[0,1],'k--',linewidth=3)


plt.axis('square')
plt.grid(linestyle='--',color=[.8,.8,.8])
plt.xlim([-7,7])
plt.ylim([-7,7])
plt.legend()
plt.savefig('Figure_03_04.png',dpi=300)
plt.show()



## Note: Make sure you run the code earlier to create variables l1, l2, etc.

# organized into lists
scalars = [ l1,l2,l3 ]
vectors = [ v1,v2,v3 ]

# initialize the linear combination
linCombo = np.zeros(len(v1))

# implement linear weighted combination using zip()
for s,v in zip(scalars,vectors):
  linCombo += s*v

# confirm it's the same answer as above
linCombo



# Whether it works or throws an error depends on how you set up the code.
# Using zip() as above works, because zip() will use only the minimum-matching items.
# Re-writing the code using indexing will cause an error, as in the code below.

# make the scalars longer
scalars = [ l1,l2,l3,5 ]
vectors = [ v1,v2,v3 ]

linCombo = np.zeros(len(v1))

for i in range(len(scalars)):
  linCombo += scalars[i]*vectors[i]



# the vector set (just one vector)
A  = np.array([ 1,3 ])

# x-axis range
xlim = [-4,4]

# random scalars in that range
scalars = np.random.uniform(low=xlim[0],high=xlim[1],size=100)



# create a figure, etc
plt.figure(figsize=(6,6))

# loop over the random scalars
for s in scalars:

  # create point p
  p = A*s

  # plot it
  plt.plot(p[0],p[1],'ko')


plt.xlim(xlim)
plt.ylim(xlim)
plt.grid()
plt.text(-4.5,4.5,'A)',fontweight='bold',fontsize=18)
plt.savefig('Figure_03_07a.png',dpi=300)
plt.show()

import plotly.graph_objects as go

# two vectors in R3
v1 = np.array([ 3,5,1 ])
v2 = np.array([ 0,2,2 ])

## uncomment the lines below for the second part
# v1 = np.array([ 3.0,5.0,1.0 ])
# v2 = np.array([ 1.5,2.5,0.5 ])


# random scalars in the x-axis range
scalars = np.random.uniform(low=xlim[0],high=xlim[1],size=(100,2))



# create random points
points = np.zeros((100,3))
for i in range(len(scalars)):

  # define this point as a random weighted combination of the two vectors
  points[i,:] = v1*scalars[i,0] + v2*scalars[i,1]


# draw the dots in the plane
fig = go.Figure( data=[go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], 
                                    mode='markers', marker=dict(size=6,color='black') )])

fig.update_layout(margin=dict(l=0,r=0,b=0,t=0))
plt.savefig('Figure_03_07b.png',dpi=300)
fig.show()


