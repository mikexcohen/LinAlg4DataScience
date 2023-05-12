import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec # for the subplots

import pandas as pd
import seaborn as sns

# NOTE: these lines define global figure properties used for publication.
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg') # display figures in vector format
plt.rcParams.update({'font.size':14}) # set global font size



# Create some correlated data
X = np.random.randn(1000,2)
X[:,1] = np.sum(X,axis=1)

# quick PCA
evals,evecs = np.linalg.eig( np.cov(X.T,ddof=1) )
scores = X @ evecs


# show in a plot
_,axs = plt.subplots(1,2,figsize=(10,5))
axs[0].plot(X[:,0],X[:,1],'ko',markerfacecolor='w')
axs[0].plot([0,3*evecs[0,1]],[0,3*evecs[1,1]],'r-',linewidth=4,label='PC1')
axs[0].plot([0,3*evecs[0,0]],[0,3*evecs[1,0]],'r:',linewidth=4,label='PC2')
axs[0].axis([-5,5,-5,5])
axs[0].set_xlabel('Data axis 1')
axs[0].set_ylabel('Data axis 2')
axs[0].legend()
axs[0].set_title('Data in channel space')


axs[1].plot(scores[:,1],scores[:,0],'ko',markerfacecolor='w')
axs[1].set_xlabel('PC axis 1')
axs[1].set_ylabel('PC axis 2')
axs[1].axis([-6,6,-6,6])
axs[1].set_title('Data in PC space')

plt.tight_layout()
plt.savefig('Figure_15_01.png',dpi=300)
plt.show()

# Empirical demonstration that variance and squared vector norm are equal.
# You can prove their equivalence by writing down their formulas and assuming the vector is mean-centered.

# extract one variable
q = X[:,1]

# compute variance
var = np.var(q,ddof=1)

# compute squared vector norm (after mean-centering)
norm = np.linalg.norm( q-np.mean(q) )**2

# show that they're the same (with the scaling factor)
print(var)
print(norm / (len(q)-1))



# Data citation: Akbilgic, Oguz. (2013). ISTANBUL STOCK EXCHANGE. UCI Machine Learning Repository.
# data source website: https://archive-beta.ics.uci.edu/ml/datasets/istanbul+stock+exchange

# import the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00247/data_akbilgic.xlsx"
data = pd.read_excel(url,index_col=0,skiprows=1)

# let's have a look
data

# show some data in line plots
data.plot(figsize=(15,6),ylabel='Market returns')
plt.savefig('Figure_15_03a.png',dpi=300)
plt.show()

# Seaborn's pairplot shows a lot of positive correlations
# I don't show this in the book b/c it's too big, lol.
sns.pairplot(data,height=1.5)
plt.show()

### show the correlation matrix in an image

plt.figure(figsize=(8,8))
heatmap = sns.heatmap(data.corr(),vmin=-1,vmax=1,annot=True,cmap='bwr')
plt.savefig('Figure_15_03b.png',dpi=300)
plt.show()



#### now for PCA!

# Step 1: covariance matrix
X = data.values # extract data
X = X - np.mean(X,axis=0,keepdims=True) # mean-center via broadcasting

# note: these data are observations-by-features, so we need X'X, not XX'
covmat = X.T@X / (X.shape[0]-1)

# visualize it
plt.figure(figsize=(6,6))
plt.imshow(covmat,vmin=-.0002,vmax=.0002)
plt.colorbar(shrink=.82)
plt.title('Data covariance')
plt.xticks(range(X.shape[1]),labels=data.columns,rotation=90)
plt.yticks(range(X.shape[1]),labels=data.columns)
plt.savefig('Figure_15_03c.png',dpi=300)
plt.show()

# Step 2: eigendecomposition
evals,evecs = np.linalg.eig(covmat)

# Step 3: sort results
sidx  = np.argsort(evals)[::-1]
evals = evals[sidx]
evecs = evecs[:,sidx]


# Step 4: component scores
components = data.values @ evecs[:,0:2]
print(components.shape)

# Step 5: eigenvalues to %var
factorScores = 100*evals/np.sum(evals)


# show scree plot
plt.figure(figsize=(8,4))
plt.plot(factorScores,'ks-',markersize=15)
plt.xlabel('Component index')
plt.ylabel('Percent variance')
plt.title('Scree plot of stocks dataset')
plt.grid()
plt.show()

# Show that variance of the components equals the eigenvalue
print('Variance of first two components:')
print(np.var(components,axis=0,ddof=1)) # note the ddof=1! The default produces the biased variance.

print(f'\nFirst two eigenvalues:')
print(evals[:2])

# correlate first two components

plt.figure(figsize=(12,6))
plt.plot(components)
plt.xlabel('Time (day)')
plt.legend(['Comp. 1','Comp. 2'])
plt.title(f'Correlation r={np.corrcoef(components.T)[0,1]:.5f}')
plt.show()

_,axs = plt.subplots(1,2,figsize=(12,5))

for i in range(2):
  axs[i].bar(range(X.shape[1]),evecs[:,i],color='black')
  axs[i].set_xticks(range(X.shape[1]))
  axs[i].set_xticklabels(data.columns,rotation=45)
  axs[i].set_ylabel('Weight')
  axs[i].set_title(f'Weights for component {i}')

plt.tight_layout()
plt.show()

# Now all in one figure

fig = plt.figure(figsize=(10,6))
gs = GridSpec(2,4,figure=fig)

# scree plot
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(factorScores,'ks-',markersize=10)
ax1.set_xlabel('Component index')
ax1.set_ylabel('Percent variance')
ax1.set_title('Scree plot')
ax1.grid()


# component time series
ax2 = fig.add_subplot(gs[0,1:])
ax2.plot(components)
ax2.set_xlabel('Time (day)')
ax2.set_xlim([0,components.shape[0]])
ax2.legend(['Comp. 1','Comp. 2'])
ax2.set_title(f'Correlation r={np.corrcoef(components.T)[0,1]:.5f}')


# bar plots of component loadings
axs = fig.add_subplot(gs[1,:2]), fig.add_subplot(gs[1,2:])
for i in range(2):
  axs[i].bar(range(X.shape[1]),evecs[:,i],color='black')
  axs[i].set_xticks(range(X.shape[1]))
  axs[i].set_xticklabels(data.columns,rotation=45)
  axs[i].set_ylabel('Weight')
  axs[i].set_title(f'Weights for component {i}')


plt.tight_layout()
plt.savefig('Figure_15_04.png',dpi=300)
plt.show()



### SVD on covariance matrix

# It suffices to show that the eigenvalues and singular values match, and that the eigenvectors and singular vectors match.
# Here I only show the first four values and the first vector.

# SVD
U,s,Vt = np.linalg.svd(covmat)

# eigen/singular values
print('First 4 eigenvalues:')
print(evals[:4])

print(f'\nFirst 4 singular values:')
print(s[:4])


# eigen/singular vectors
print('\n\n\nFirst eigenvector:')
print(evecs[:,0])

print('\nFirst singular vector:')
print(U[:,0])

### SVD on data matrix

# Again, we can simply show that the singular values (suitably normalized) match the eigenvalues, and that
# the singular vectors match the eigenvectors.

# Note that the data variable X is already mean-centered!
U,s,Vt = np.linalg.svd(X)  # SVD


# eigen/singular values
print('First 4 eigenvalues:')
print(evals[:4])

print(f'\nFirst 4 singular values:')
print(s[:4]**2/(X.shape[0]-1))


# eigen/singular vectors
print('\n\n\nFirst eigenvector:')
print(evecs[:,0])

print('\nFirst right singular vector:')
print(Vt[0,:])



### As above, it suffices to show that the eigenvalues and eigenvectors match.

from sklearn.decomposition import PCA
 
pca = PCA()
X_t = pca.fit_transform(data)

# compare percent-normalized eigenvalues
print('Eigenvalues:')
print(evals[:4])

print(f'\nExplained variance from sklearn:')
print(pca.explained_variance_[:4])



# eigenvector and sklearn component
print('\n\n\nFirst eigenvector:')
print(evecs[:,0])

print('\nFirst sklearn component vector:')
print(pca.components_[0,:])



# generate data

x = np.hstack((np.random.randn(1000,1),.05*np.random.randn(1000,1)))

# rotation matrices
th = -np.pi/6
R1 = np.array([ [np.cos(th), -np.sin(th)],
                [np.sin(th),  np.cos(th)] ])
th = -np.pi/3
R2 = np.array([ [np.cos(th), -np.sin(th)],
                [np.sin(th),  np.cos(th)] ])

# create the data
X = np.vstack((x@R1,x@R2))
X.shape

# PCA via SVD
U,s,Vt = np.linalg.svd(X-np.mean(X,axis=0,keepdims=True))

# not necessary: convert singular values into eigenvalues
s = s**2 / (X.shape[0]-1)

# also not necessary: up-scale the singular vectors for visualization
Vt *= 2

# plot the data and eigenvectors

plt.figure(figsize=(7,7))

# the data
plt.plot(X[:,0],X[:,1],'ko',markerfacecolor='w')

# eigenvectors
plt.plot([0,Vt[0,0]],[0,Vt[1,0]],'r--',linewidth=5,label='Comp 1')
plt.plot([0,Vt[0,1]],[0,Vt[1,1]],'r:',linewidth=5,label='Comp 2')

plt.legend()
plt.grid()
plt.savefig('Figure_15_05.png',dpi=300)
plt.show()



# create the data
N = 200

class1 = np.random.randn(N,2)
class1[:,1] += class1[:,0]
class1 += np.array([2,-1])

class2 = np.random.randn(N,2)
class2[:,1] += class2[:,0]

# for later, it will be convenient to have the data in one matrix
alldata = np.vstack((class1,class2))
labels  = np.append(np.zeros(N),np.ones(N))



# show data in their original data space
ax = sns.jointplot(x=alldata[:,0],y=alldata[:,1],hue=labels)
ax.ax_joint.set_xlabel('Data axis 1')
ax.ax_joint.set_ylabel('Data axis 2')
ax.plot_joint(sns.kdeplot)
plt.savefig('Figure_15_02a.png',dpi=300)
plt.show()



# LDA

# between-class covariance
cmc1 = np.mean(class1,axis=0)
cmc2 = np.mean(class2,axis=0)
covB = np.cov(np.vstack((cmc1,cmc2)).T,ddof=1)

# within-class covariances
cov1 = np.cov(class1.T,ddof=1)
cov2 = np.cov(class2.T,ddof=1)
covW = (cov1+cov2)/2


# LDA via GED
from scipy.linalg import eigh
evals,evecs = eigh(covB,covW)

# sort the solution
sidx  = np.argsort(evals)[::-1]
evals = evals[sidx]
evecs = evecs[:,sidx]


# project the mean-centered data onto the GED axes
projA = (alldata-np.mean(alldata,axis=0)) @ evecs  # A=all

# show the data
_,axs = plt.subplots(1,2,figsize=(12,6))
marker = ['bo','r+']
for i in range(2):
  axs[0].plot(alldata[labels==i,0],alldata[labels==i,1],marker[i],label=f'Class {i}')

axs[0].plot([0,evecs[0,0]],[0,evecs[1,0]],'k-',linewidth=3,label='C1')
axs[0].plot([0,evecs[0,1]],[0,evecs[1,1]],'k:',linewidth=3,label='C2')
axs[0].set_xlabel('Data axis 1')
axs[0].set_ylabel('Data axis 2')
axs[0].set_title('Data in variable space')



# and again in the GED space
for i in range(2):
  axs[1].plot(projA[labels==i,0],projA[labels==i,1],marker[i],label=f'Class {i}')
axs[1].set_xlabel('GED axis 1')
axs[1].set_ylabel('GED axis 2')
axs[1].set_title('Data in GED space')


# common settings
for i in range(2):
  axs[i].axis([-6,6,-6,6])
  axs[i].grid()
  axs[i].legend()

plt.tight_layout()
plt.savefig('Figure_15_06ab.png',dpi=300)
plt.show()

# prediction (converted to ints)
predictedLabel = ( projA[:,0] > 0 )+0

print(f'Prediction accuracy: {100*np.mean( predictedLabel==labels )}%')

# show the results
plt.figure(figsize=(12,5))
plt.plot(predictedLabel,'ks',markersize=7,markerfacecolor='w',linewidth=2)
plt.plot([N-.5,N-.5],[-.5,1.5],'k--')
plt.xlabel('Sample number')
plt.ylabel('Predicted class')
plt.yticks([0,1],labels=['Class 0','Class 1'])
plt.title(f'Accuracy = {100*np.mean(predictedLabel==labels):.2f}%')
plt.savefig('Figure_15_06c.png',dpi=300)
plt.show()

# redraw the jointplot in the GED space (used in Figure 2)
ax = sns.jointplot(x=projA[:,0],y=projA[:,1],hue=labels,xlim=[-6,6],ylim=[-6,6])
ax.ax_joint.set_xlabel('LDA axis 1')
ax.ax_joint.set_ylabel('LDA axis 2')
ax.plot_joint(sns.kdeplot)
plt.savefig('Figure_15_02b.png',dpi=300)
plt.show()



# not the identity matrix!
print("V'V:")
print(np.round( evecs.T @ evecs ,3))


# yes the identity matrix!
print(f"\nV'RV:")
print(np.round( evecs.T @ covW @ evecs ,3))



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

ldamodel = LDA(solver='eigen')
ldamodel.fit(alldata,labels)


# show the results
plt.figure(figsize=(12,5))
plt.plot(predictedLabel,'ks',markersize=7,markerfacecolor='w',linewidth=2,label='My LDA')
plt.plot(ldamodel.predict(alldata),'r+',markersize=10,markerfacecolor='w',linewidth=2,label='sklearn LDA')
plt.plot([N-.5,N-.5],[-.5,1.5],'k--')
plt.xlabel('Sample number')
plt.ylabel('Predicted class')
plt.yticks([0,1],labels=['Class 0','Class 1'])
plt.ylim([-.5,1.5])
plt.legend()
plt.title(f'Accuracy = {100*np.mean(ldamodel.predict(alldata)==labels):.2f}%')
plt.savefig('Figure_15_07.png',dpi=300)
plt.show()



# shrinkage amounts
shrinkage = np.linspace(0,1,21)
accuracies = np.zeros(len(shrinkage))

# loop over shrinkages and compute model accuracy
for i,s in enumerate(shrinkage):
  
  # setup the model
  ldamodel = LDA(solver='eigen',shrinkage=s)

  tmpacc = []
  for _ in range(50):

    # randomly split the data into train/test
    randorder = np.random.permutation(alldata.shape[0])

    # fit the model on the training data
    ldamodel.fit(alldata[randorder[:350],:],labels[randorder[:350]])

    # grab accuracy
    tmpacc.append(100*np.mean(ldamodel.predict(alldata[randorder[350:],:])==labels[randorder[350:]]))

  # evaluate model performance on the test data
  accuracies[i] = np.mean(tmpacc)


# plot!
plt.figure(figsize=(8,5))
plt.plot(shrinkage,accuracies,'ks-',markersize=10,markerfacecolor='w',linewidth=2)
plt.xlabel('Shrinkage amount')
plt.ylabel('Prediction accuracy on validation trials')
plt.title('Effect of shrinkage on model performance')
plt.savefig('Figure_15_08.png',dpi=300)
plt.show()



from skimage import io,color
url = 'https://upload.wikimedia.org/wikipedia/en/1/1c/Stravinsky_picasso.png'

# import picture and downsample to 2D
strav = io.imread(url)
strav = color.rgb2gray(strav)

plt.figure(figsize=(8,8))
plt.imshow(strav,cmap='gray')
plt.title(f'Matrix size: {strav.shape}, rank: {np.linalg.matrix_rank(strav)}')
plt.show()

# SVD
U,s,Vt = np.linalg.svd(strav)
S = np.zeros_like(strav)
np.fill_diagonal(S,s)

# show scree plot
plt.figure(figsize=(12,4))
plt.plot(s[:30],'ks-',markersize=10)
plt.xlabel('Component index')
plt.ylabel('Singular value')
plt.title('Scree plot of Stravinsky picture')
plt.grid()
plt.show()

fig = plt.figure(figsize=(9,9))
gs = GridSpec(3,4,figure=fig)

# the image
ax1 = fig.add_subplot(gs[0,0])
ax1.imshow(strav,cmap='gray')
ax1.set_title(f'Matrix size: {strav.shape},\nrank: {np.linalg.matrix_rank(strav)}')

# scree plot
ax2 = fig.add_subplot(gs[0,1:])
ax2.plot(s[:30],'ks-',markersize=10)
ax2.set_xlabel('Component index')
ax2.set_ylabel('Singular value')
ax2.set_title('Scree plot of Stravinsky picture')
ax2.grid()


## now show the first N "layers" separately
numLayers = 4
rank1mats = np.zeros((numLayers,strav.shape[0],strav.shape[1]))


# the loop
for i in range(numLayers):
    
    # create this layer
    rank1mats[i,:,:] = np.outer(U[:,i],Vt[i,:])*s[i]
    
    # show this layer
    ax = fig.add_subplot(gs[1,i])
    ax.imshow(rank1mats[i,:,:],cmap='gray')
    ax.set_title(f'L {i}')
    ax.set_xticks([]), ax.set_yticks([])

    # show the cumulative sum of layers
    ax = fig.add_subplot(gs[2,i])
    ax.imshow(np.sum(rank1mats[:i+1,:,:],axis=0),cmap='gray')
    ax.set_title(f'L 0:{i}')
    ax.set_xticks([]), ax.set_yticks([])


plt.tight_layout()
plt.savefig('Figure_15_09.png',dpi=300)
plt.show()



# Reconstruct based on first k layers

# number of components
k = 80

# reconstruction
stravRec = U[:,:k] @ S[:k,:k] @ Vt[:k,:]


# show the original, reconstructed, and error
_,axs = plt.subplots(1,3,figsize=(15,6))

axs[0].imshow(strav,cmap='gray',vmin=.1,vmax=.9)
axs[0].set_title('Original')

axs[1].imshow(stravRec,cmap='gray',vmin=.1,vmax=.9)
axs[1].set_title(f'Reconstructed (k={k}/{len(s)})')

axs[2].imshow((strav-stravRec)**2,cmap='gray',vmin=0,vmax=1e-1)
axs[2].set_title('Squared errors')

plt.tight_layout()
plt.savefig('Figure_15_10.png',dpi=300)
plt.show()

# compute sizes of the images
stravSize  = strav.nbytes / 1024**2
stravRSize = stravRec.nbytes / 1024**2

# and of the vectors/values
uSize = U[:,:k].nbytes / 1024**2
sSize = s[:k].nbytes / 1024**2
vSize = Vt[:k,:].nbytes / 1024**2


# print image sizes
print(f'      Original is {stravSize:.2f} mb')
print(f'Reconstruction is {stravRSize:.2f} mb')
print(f'Recon vectors are {uSize+sSize+vSize:.2f} mb (using k={k} comps.)')

print(f'\nCompression of {100*(uSize+sSize+vSize)/stravSize:.2f}%')



# range of components
k = range(1,len(s)+1)

# initialize variable to store results
kError = np.zeros(len(k))


# the loop
for i in range(len(k)):
  
  # reconstruction
  stravRec = U[:,:k[i]] @ S[:k[i],:k[i]] @ Vt[:k[i],:]

  # compute and store the error
  kError[i] = np.sqrt(np.sum((strav-stravRec)**2))



# show the results
plt.figure(figsize=(10,7))
plt.plot(k,kError,'ks-')
# plt.plot(k[:-1],np.diff(kError),'ks-') # uncomment to show derivative (and comment out the previous line)
plt.xlabel('Rank of reconstruction')
plt.ylabel('Error from original')
plt.title('Reconstruction accuracy')
plt.savefig('Figure_15_11.png',dpi=300)
plt.show()



# create a spatial sine wave

# sine phases
sinefreq = .02   # arbitrary units
sinephas = np.pi/6 # rotate

# sine wave initializations
[x,y] = np.meshgrid(np.linspace(-100,100,strav.shape[1]),
                    np.linspace(-100,100,strav.shape[0]))
xp    = x*np.cos(sinephas) + y*np.sin(sinephas)


# compute sine wave
sinimg = np.sin( 2*np.pi*sinefreq*xp)

# scale to [0 1]
sinimg = (sinimg-np.min(sinimg)) / (np.max(sinimg)-np.min(sinimg))


# add to stravinsky picture and re-scale (using two lines)
stravNoise = strav + sinimg
stravNoise = stravNoise-np.min(stravNoise)
stravNoise = stravNoise/np.max(stravNoise)

# let's see it!
_,axs = plt.subplots(1,3,figsize=(10,7))
axs[0].imshow(strav,cmap='gray')
axs[0].set_title('Original picture')

axs[1].imshow(sinimg,cmap='gray')
axs[1].set_title('Noise image')

axs[2].imshow(stravNoise,cmap='gray')
axs[2].set_title('Contaminated picture')

plt.tight_layout()
plt.savefig('Figure_15_12.png',dpi=300)
plt.show()

# SVD
Un,sn,Vtn = np.linalg.svd(stravNoise)
Sn = np.zeros_like(stravNoise)
np.fill_diagonal(Sn,sn)

# show scree plot
plt.figure(figsize=(12,4))
plt.plot(sn[:30],'ks-',markersize=10)
plt.xlabel('Component index')
plt.ylabel('Singular value')
plt.title('Scree plot of Noisy Stravinsky picture')
plt.grid()
plt.show()

fig = plt.figure(figsize=(9,9))
gs = GridSpec(3,4,figure=fig)

# the image
ax1 = fig.add_subplot(gs[0,0])
ax1.imshow(stravNoise,cmap='gray')
ax1.set_title(f'Matrix size: {strav.shape},\nrank: {np.linalg.matrix_rank(stravNoise)}')

# scree plot
ax2 = fig.add_subplot(gs[0,1:])
ax2.plot(sn[:30],'ks-',markersize=10)
ax2.set_xlabel('Component index')
ax2.set_ylabel('Singular value')
ax2.set_title('Scree plot of noisy Stravinsky picture')
ax2.grid()


## now show the first N "layers" separately
numLayers = 4
rank1mats = np.zeros((numLayers,strav.shape[0],strav.shape[1]))


# the loop
for i in range(numLayers):
    
    # create this layer
    rank1mats[i,:,:] = np.outer(Un[:,i],Vtn[i,:])*sn[i]
    
    # show this layer
    ax = fig.add_subplot(gs[1,i])
    ax.imshow(rank1mats[i,:,:],cmap='gray')
    ax.set_title(f'L {i}')
    ax.set_xticks([]), ax.set_yticks([])

    # show the cumulative sum of layers
    ax = fig.add_subplot(gs[2,i])
    ax.imshow(np.sum(rank1mats[:i+1,:,:],axis=0),cmap='gray')
    ax.set_title(f'L 0:{i}')
    ax.set_xticks([]), ax.set_yticks([])


plt.tight_layout()
plt.savefig('Figure_15_13.png',dpi=300)
plt.show()



# Reconstruct based on first k layers

# noise components
noiseComps = np.array([1,2])

# reconstruction of the noise
stravRecNoise = Un[:,noiseComps] @ Sn[noiseComps,:][:,noiseComps] @ Vtn[noiseComps,:]


# reconstruction of the image with noise projected out
noNoiseCompsU = np.full(Un.shape[0],True)
noNoiseCompsU[noiseComps] = False

noNoiseCompsV = np.full(Vtn.shape[0],True)
noNoiseCompsV[noiseComps] = False

# here's the image without the noise components
stravRecNoNoise = Un[:,noNoiseCompsU] @ Sn[noNoiseCompsU,:][:,noNoiseCompsV] @ Vtn[noNoiseCompsV,:]




# show the original, reconstructed, and error
_,axs = plt.subplots(1,3,figsize=(15,6))

axs[0].imshow(stravNoise,cmap='gray')
axs[0].set_title('Noisy image')

axs[1].imshow(stravRecNoise,cmap='gray',vmin=-.5,vmax=.5)
axs[1].set_title(f'Only noise (comps {noiseComps})')

axs[2].imshow(stravRecNoNoise,cmap='gray',vmin=.1,vmax=.9)
axs[2].set_title('Noise projected out')

plt.tight_layout()
plt.savefig('Figure_15_14.png',dpi=300)
plt.show()

# histogram of noise reconstruction
plt.hist(stravRecNoise.flatten(),100);


