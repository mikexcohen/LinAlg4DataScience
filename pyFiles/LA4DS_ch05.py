import numpy as np
import matplotlib.pyplot as plt

# for null spaces
import scipy.linalg

# a pretty-looking matrix from scipy
from scipy.linalg import toeplitz


# NOTE: these lines define global figure properties used for publication.
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg') # print figures in svg format
plt.rcParams.update({'font.size':14}) # set global font size

v = np.array([[1,2,3]]).T # col vector
w = np.array([[10,20]])   # row vector
v + w

# create some matrices
A = np.random.randn(3,4)
B = np.random.randn(100,100)
C = -toeplitz(np.arange(8),np.arange(10))


# and show them as images
fig,axs = plt.subplots(1,3,figsize=(10,3))

axs[0].imshow(A,cmap='gray')
axs[1].imshow(B,cmap='gray')
axs[2].imshow(C,cmap='gray')

for i in range(3): axs[i].axis('off')
plt.tight_layout()
plt.savefig('Figure_05_01.png',dpi=300)
plt.show()



# create a matrix
A = np.reshape(np.arange(1,10),(3,3))
print(A)

# get the n-th row

print( A[1,:] )

# note that to extract only one row, you don't need the column indices. 
print( A[1] )
# But that's potentially confusing, so I recommend avoiding that notation.

# get the n-th column
print( A[:,1] )
# Note that it prints out as a "row" even thought it's a column of the matrix

# multiple rows
A[0:2,:]

# multiple columns
A[:,1:]

## extracting a submatrix (multiple rows and cols)

# The goal here is to extract a submatrix from matrix A. Here's A:
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

# And we want rows 0-1 and columns 0-1, thus:
# [[1 2]
#  [4 5]]


# seems like this should work...
print( A[0:2,1:2] )
print(' ')

# but this does (remember x:y:z slices from x to y-1 in steps of z)
print( A[0:2:1,0:2:1] )

# This cell has the example shown in the book.

# the full matrix
A = np.arange(60).reshape(6,10)

# a block of it
sub = A[1:4:1,0:5:1]


# print them out
print('Original matrix:\n')
print(A)

print('\n\nSubmatrix:\n')
print(sub)



## create some matrices

# square
M1 = np.random.permutation(16).reshape(4,4)

# upper-triangular square
M2 = np.triu(np.random.randint(10,20,(3,3)))

# lower-triangular rectangular
M3 = np.tril(np.random.randint(8,16,(3,5)))

# diagonal
M4 = np.diag( np.random.randint(0,6,size=8) )

# identity
M5 = np.eye(4,dtype=int)

# zeros
M6 = np.zeros((4,5),dtype=int)

matrices  = [ M1,M2,M3,M4,M5,M6 ]
matLabels = [ 'Square','Upper-triangular','Lower-triangular','Diagonal','Identity','Zeros'  ]


_,axs = plt.subplots(2,3,figsize=(12,6))
axs = axs.flatten()

for mi,M in enumerate(matrices):
  axs[mi].imshow(M,cmap='gray',origin='upper',
                 vmin=np.min(M),vmax=np.max(M))
  axs[mi].set(xticks=[],yticks=[])
  axs[mi].set_title(matLabels[mi])
  
  # text labels
  for (j,i),num in np.ndenumerate(M):
    axs[mi].text(i,j,num,color=[.8,.8,.8],ha='center',va='center',fontweight='bold')



plt.savefig('Figure_05_02.png',dpi=300)
plt.tight_layout()
plt.show()



# matrix size parameters (called 'shape' in Python lingo)
Mrows = 4 # shape 0
Ncols = 6 # shape 1

# create the matrix!
A = np.random.randn(Mrows,Ncols)

# print out the matrix (rounding to facilitate visual inspection)
np.round(A,3)

# Extract the triangular part of a dense matrix

M = 4
N = 6
A = np.random.randn(M,N)

# upper triangular
print('Upper triangular:\n')
print(np.triu(A))

# lower triangular
print('\n\nLower triangular:\n')
print(np.tril(A))

# Diagonal

# input a matrix to get the diagonal elements
A = np.random.randn(5,5)
d = np.diag(A)
print('Input a matrix:\n',d)

# OR input a vector to create a diagonal matrix!
v = np.arange(1,6)
D = np.diag(v)
print('\n\nInput a vector:\n',D)

# Identity and zeros matrices

# Note that you only specify one input
n = 4
I = np.eye(n)
print(f'The {n}x{n} identity matrix:\n',I)


# Zeros matrix

# Important: All shape parameters are given as one input (a tuple or list),
#            unlike np.random.randn()
n = 4
m = 5
I = np.zeros((n,m))
print(f'The {n}x{m} zeros matrix:\n',I)



A = np.array([  [2,3,4],
                [1,2,4] ])

B = np.array([  [ 0, 3,1],
                [-1,-4,2] ])

print(A+B)



# Not shifting; broadcasting scalar addition
3 + np.eye(2)

# This is shifting:

# the matrix
A = np.array([ [4,5, 1],
               [0,1,11],
               [4,9, 7]  ])

# the scalar
s = 6

print('Original matrix:')
print(A), print(' ')

# as in the previous cell, this is broadcasting addition, not shifting
print('Broadcasting addition:')
print(A + s), print(' ')

# This is shifting
print('Shifting:')
print( A + s*np.eye(len(A)) )



print(A), print(' ')
print(s*A)



# two random matrices
A = np.random.randn(3,4)
B = np.random.randn(3,4)

# this is Hadamard multiplication
A*B 

# and so is this
np.multiply(A,B)

# this one is NOT Hadamard multiplication
A@B



# Create a few matrices
A = np.random.randn(3,6)
B = np.random.randn(6,4)
C = np.random.randn(6,4)

# try some multiplications, and print out the shape of the product matrix
print( (A@B).shape )
print( np.dot(A,B).shape ) # same as above
print( (B@C).shape )
print( (A@C).shape )

# Note/reminder:

# This is Hadamard (element-wise) multiplication:
print( np.multiply(B,C) ), print(' ')

# This is matrix multiplication
print( np.dot(B,C.T) )

# demonstration:
# np.dot(B,C.T)-B@C.T



# some matrix
M  = np.array([ [2,3],[2,1] ])
x  = np.array([ [1,1.5] ]).T # transposed into a column vector!
Mx = M@x


plt.figure(figsize=(6,6))

plt.plot([0,x[0,0]],[0,x[1,0]],'k',linewidth=4,label='x')
plt.plot([0,Mx[0,0]],[0,Mx[1,0]],'--',linewidth=3,color=[.7,.7,.7],label='Mx')
plt.xlim([-7,7])
plt.ylim([-7,7])
plt.legend()
plt.grid()
plt.savefig('Figure_05_05a.png',dpi=300)
plt.show()

# some matrix
M  = np.array([ [2,3],[2,1] ])
v  = np.array([ [1.5,1] ]).T # transposed into a column vector!
Mv = M@v


plt.figure(figsize=(6,6))

plt.plot([0,v[0,0]],[0,v[1,0]],'k',linewidth=4,label='v')
plt.plot([0,Mv[0,0]],[0,Mv[1,0]],'--',linewidth=3,color=[.7,.7,.7],label='Mv')
plt.xlim([-7,7])
plt.ylim([-7,7])
plt.legend()
plt.grid()
plt.savefig('Figure_05_05b.png',dpi=300)
plt.show()





# A matrix to transpose
A = np.array([ [3,4,5],[1,2,3] ])

A_T1 = A.T # as method
A_T2 = np.transpose(A) # as function

# double-transpose
A_TT = A_T1.T 


# print them
print( A_T1 ), print(' ')
print( A_T2 ), print(' ')
print( A_TT )





# indexing

A = np.arange(12).reshape(3,4)
print(A)

# find the element in the 2nd row, 4th column
ri = 1
ci = 3

print(f'The matrix element at index ({ri+1},{ci+1}) is {A[ri,ci]}')

# Create the matrix
C = np.arange(100).reshape((10,10))

# extract submatrix
C_1 = C[0:5:1,0:5:1]

# here's what the matrices look like
print(C), print(' ')
print(C_1)



# visualize the matrices as maps
_,axs = plt.subplots(1,2,figsize=(10,5))

axs[0].imshow(C,cmap='gray',origin='upper',vmin=0,vmax=np.max(C))
axs[0].plot([4.5,4.5],[-.5,9.5],'w--')
axs[0].plot([-.5,9.5],[4.5,4.5],'w--')
axs[0].set_title('Original matrix')
# text labels
for (j,i),num in np.ndenumerate(C):
  axs[0].text(i,j,num,color=[.8,.8,.8],ha='center',va='center')


axs[1].imshow(C_1,cmap='gray',origin='upper',vmin=0,vmax=np.max(C))
axs[1].set_title('Submatrix')
# text labels
for (j,i),num in np.ndenumerate(C_1):
  axs[1].text(i,j,num,color=[.8,.8,.8],ha='center',va='center')


plt.savefig('Figure_05_06.png',dpi=300)
plt.show()

# cut it into blocks
C_1 = C[0:5:1,0:5:1]
C_2 = C[0:5:1,5:10:1]
C_3 = C[5:10:1,0:5:1]
C_4 = C[5:10:1,5:10:1]

# rearrange the blocks
newMatrix = np.vstack( (np.hstack((C_4,C_3)),
                        np.hstack((C_2,C_1))) )


# visualize the matrices
_,axs = plt.subplots(1,2,figsize=(10,5))

axs[0].imshow(C,cmap='gray',origin='upper',vmin=0,vmax=np.max(C))
axs[0].plot([4.5,4.5],[-.5,9.5],'w--')
axs[0].plot([-.5,9.5],[4.5,4.5],'w--')
axs[0].set_title('Original matrix')
# text labels
for (j,i),num in np.ndenumerate(C):
  axs[0].text(i,j,num,color=[.8,.8,.8],ha='center',va='center')


axs[1].imshow(newMatrix,cmap='gray',origin='upper',vmin=0,vmax=np.max(C))
axs[1].plot([4.5,4.5],[-.5,9.5],'w--')
axs[1].plot([-.5,9.5],[4.5,4.5],'w--')
axs[1].set_title('Block-shifted')
# text labels
for (j,i),num in np.ndenumerate(newMatrix):
  axs[1].text(i,j,num,color=[.8,.8,.8],ha='center',va='center')

plt.savefig('Figure_05_07.png',dpi=300)
plt.show()

def addMatrices(A,B):

  # check that both matrices have the same size
  if A.shape != B.shape:
    raise('Matrices must be the same size!')

  # initialize sum matrix
  C = np.zeros(A.shape)

  # sum!
  for i in range(A.shape[0]):
    for j in range(A.shape[1]):
      C[i,j] = A[i,j] + B[i,j]
  
  return C


# test the function
M1 = np.zeros((6,4))
M2 = np.ones((6,4))

addMatrices(M1,M2)


# create random matrices and a scalar
A = np.random.randn(3,4)
B = np.random.randn(3,4)
s = np.random.randn()

# equations shown in the text
expr1 = s*(A+B)
expr2 = s*A + s*B
expr3 = A*s + B*s


# There are a few ways to test for 3-way equality. 
# My choice below is that if x=y=z, then 2x-y-z=0.

# print out, rounded to 8 digits after the decimal point
print(np.round(2*expr1 - expr2 - expr3,8))



# generate two matrices
m = 4
n = 6
A = np.random.randn(m,n);
B = np.random.randn(n,m)

# build up the product matrix element-wise
C1 = np.zeros((m,m))
for rowi in range(m):
  for coli in range(m):
    C1[rowi,coli] = np.dot( A[rowi,:],B[:,coli] )
    


# implement matrix multiplication directly
C2 = A@B

# compare the results (using isclose(); results should be a matrix of TRUEs)
np.isclose( C1,C2 )



# Create the matrices
L = np.random.randn(2,6)
I = np.random.randn(6,3)
V = np.random.randn(3,5)
E = np.random.randn(5,2)

# multiplications indicated in the instructions
res1 = ( L@I@V@E ).T
# res2 = L.T @ I.T @ V.T @ E.T
res3 = E.T @ V.T @ I.T @ L.T

# show that res1 and res3 are the same (within rounding error tolerance)
print(res1-res3)



def isMatrixSymmetric(S):
  
  # difference between matrix and its transpose
  D = S-S.T

  # check whether sum of squared errors (SSE) is smaller than a threshold
  sse = np.sum(D**2)

  # output TRUE if sse is tiny; FALSE means the matrix is asymmetric
  return sse<10**-15

# note: There are many other ways you could solve this. 
# If you want to explore different methods, consider np.all() or np.isclose()

# create symmetric and nonsymmetric matrices
A = np.random.randn(4,4)
AtA = A.T@A

# test!
print(isMatrixSymmetric(A))
print(isMatrixSymmetric(AtA))



# create symmetric and nonsymmetric matrices
A = np.random.randn(4,4)
AtA = (A + A.T) / 2 # additive method!

# test!
print(isMatrixSymmetric(A))
print(isMatrixSymmetric(AtA))



import plotly.graph_objects as go

# As a matrix with two columns in R3, instead of two separate vectors
A = np.array( [ [3,0],
                [5,2],
                [1,2] ] )

# uncomment the line below
# A = np.array( [ [3,1.5],
#                 [5,2.5],
#                 [1, .5] ] )


xlim = [-4,4]
scalars = np.random.uniform(low=xlim[0],high=xlim[1],size=(100,2))

# create random points
points = np.zeros((100,3))
for i in range(len(scalars)):
  points[i,:] = A@scalars[i]

# draw the dots in the figure
fig = go.Figure( data=[go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers')])
fig.show()



n = 4

# create "base" matrices
O = np.ones((n,n))
D = np.diag(np.arange(1,n+1)**2)
S = np.sqrt(D)

# pre- and post-multiply
pre = D@O
pst = O@D

# and both
both = S@O@S



# print out the "base" matrices
print('Ones matrix:')
print(O), print(' ')

print('Diagonal matrix:')
print(D), print(' ')

print('Sqrt-diagonal matrix:')
print(S), print(' ')



print('Pre-multiply by diagonal:')
print(pre), print(' ')

print('Post-multiply by diagonal:')
print(pst), print(' ')

print('Pre- and post-multiply by sqrt-diagonal:')
print(both)



# Create two diagonal matrices
N = 5
D1 = np.diag( np.random.randn(N) )
D2 = np.diag( np.random.randn(N) )

# two forms of multiplication
hadamard = D1*D2
standard = D1@D2

# compare them
hadamard - standard




