import numpy as np
import matplotlib.pyplot as plt

# sympy library for RREF
import sympy as sym

# scipy for LU
import scipy.linalg


# used to create non-regular subplots
import matplotlib.gridspec as gridspec


# NOTE: these lines define global figure properties used for publication.
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg') # display figures in vector format
plt.rcParams.update({'font.size':14}) # set global font size



# generate some matrices
A = np.random.randn(4,4)
B = np.random.randn(4,4)

# solve for X
# 1) inv(A)@A@X = inv(A)@B
# 2) inv(A)@A@X = B@inv(A)

X1 = np.linalg.inv(A) @ B
X2 = B @ np.linalg.inv(A)

# residual (should be zeros matrix)
res1 = A@X1 - B
res2 = A@X2 - B

# which is correct?
print('res1:'), print(' ')
print( np.round(res1,10) ), print(' ')

print('res2:'), print(' ')
print( np.round(res2,10) )




# the augmented matrix
M = np.array([ [1,1,4],[-1/2,1,2] ])

# converted into a sympy matrix
symMat = sym.Matrix(M)
print(symMat)

# RREF
symMat.rref()[0] # just the first output to get the RREF matrix (the second output is the indices of the pivots per row)



# simple example with integers

# a matrix
A = np.array([ [2,2,4], [1,0,3], [2,1,2] ])

# its LU decomposition via scipy (please ignore the first output for now)
_,L,U = scipy.linalg.lu(A)

# print them out
print('L: ')
print(L), print(' ')

print('U: ')
print(U), print(' ')

print('A - LU: ')
print(A - L@U) # should be zeros

# matrix sizes
m = 4
n = 6

A = np.random.randn(m,n)

P,L,U = scipy.linalg.lu(A)

# show the matrices
fig,axs = plt.subplots(1,5,figsize=(13,4))

axs[0].imshow(A,vmin=-1,vmax=1)
axs[0].set_title('A')

axs[1].imshow(np.ones((m,n)),cmap='gray',vmin=-1,vmax=1)
axs[1].text(n/2,m/2,'=',ha='center',fontsize=30,fontweight='bold')
# axs[1].axis('off')

axs[2].imshow(P.T,vmin=-1,vmax=1)
axs[2].set_title(r'P$^T$')

axs[3].imshow(L,vmin=-1,vmax=1)
axs[3].set_title('L')

h = axs[4].imshow(U,vmin=-1,vmax=1)
axs[4].set_title('U')

for a in axs:
  a.axis('off')
  a.set_xlim([-.5,n-.5])
  a.set_ylim([m-.5,-.5])


fig.colorbar(h,ax=axs[-1],fraction=.05)
plt.tight_layout()
plt.savefig('Figure_10_01.png',dpi=300)
plt.show()



# Time-test!

import time

# start the timer
tic = time.time()

# run the test
for i in range(1000):
  A = np.random.randn(100,100)
  P,L,U = spla.lu(A)

# stop the timer
toc = time.time() - tic
toc # print the result in seconds



# make a reduced-rank random matrix

# sizes and rank
M = 6
N = 8
r = 3

# create the matrix
A = np.random.randn(M,r) @ np.random.randn(r,N)

# LU
P,L,U = scipy.linalg.lu(A)

# and plot
_,axs = plt.subplots(1,3,figsize=(12,7))

axs[0].imshow(A,vmin=-1,vmax=1,cmap='gray')
axs[0].set_title(f'A, rank={np.linalg.matrix_rank(A)}')

axs[1].imshow(L,vmin=-1,vmax=1,cmap='gray')
axs[1].set_title(f'L, rank={np.linalg.matrix_rank(L)}')

axs[2].imshow(U,vmin=-1,vmax=1,cmap='gray')
axs[2].set_title(f'U, rank={np.linalg.matrix_rank(U)}')

plt.tight_layout()
plt.savefig('Figure_10_02.png',dpi=300)
plt.show()

np.round(L,2)



# a matrix and its det
M = 6
A = np.random.randn(M,M)

# LU
P,L,U = scipy.linalg.lu(A)

# determinant as the product of the diagonals of U
detLU = np.prod( np.diag(U) ) * np.linalg.det(P)

# check against the det function
detNP = np.linalg.det(A)

# compare
print(detLU,detNP)
print(detLU-detNP)



# matrix sizes
m = 4
A = np.random.randn(m,m)

# LU decomposition
P,L,U = scipy.linalg.lu(A)

# inverse
invViaLU = np.linalg.inv(U) @ np.linalg.inv(L) @ P.T

# "regular" inverse
invViaInv = np.linalg.inv(A)

np.round( A@invViaLU ,10)



# The reason is that writing out the equation leads to PtP in the middle, which is the identity matrix. 
# Conceptually, it means that any row swaps are undone when multiplying by the transpose.

# create a matrix
A = np.random.randn(4,4)

# LUP
P,L,U = scipy.linalg.lu(A)

# compute AtA via LU
AtA_lu = U.T @ L.T @ L @ U

# direct computation
AtA_direct = A.T @ A

# compare to direct computation
np.round( AtA_lu - AtA_direct ,10)




