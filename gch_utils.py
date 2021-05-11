import numpy as np
import scipy.linalg as salg
from sklearn.preprocessing import KernelCenterer

def KFPS(x,nbOfLandmarks,seed=10,initialLandmark=None,listOfDiscardedPoints=None,verbose=False,kernel=True):
    nbOfFrames = x.shape[0]
    np.random.seed(seed)
    LandmarksIdx = np.zeros(nbOfLandmarks,int)
    if listOfDiscardedPoints is None:
        listOfDiscardedPoints = []
    if initialLandmark is None:
        isel = int(np.random.uniform()*nbOfFrames)
        while isel in listOfDiscardedPoints:
            isel=int(np.random.uniform()*nbOfFrames)
    else:
        isel = initialLandmark

    if kernel:
        diag = np.diag(x)
        distLineFn = lambda x, isel: (
            x[isel,isel] + diag - 2 * x[isel, :]
        )
    else:
        diag = np.sum(x**2, axis=1)
        distLineFn = lambda x, isel: (
            diag + diag[isel] - 2 * np.dot(x, x[isel])
        )

    ldist = 1e100*np.ones(nbOfFrames,float)

    LandmarksIdx[0] = isel
    nontrue = np.setdiff1d(range(nbOfFrames), listOfDiscardedPoints)

    for nsel in range(1,nbOfLandmarks):
        dmax = 0*np.ones(nbOfFrames,float)
        imax = 0
        distLine = distLineFn(x, isel)

        dsel = distLine[nontrue]

        low = dsel < ldist
        ldist[low] = dsel[low]
        larg = ldist > dmax
        dmax[larg] = ldist[larg]

        isel = dmax.argmax()
        LandmarksIdx[nsel] = isel
        if verbose is True:
            print("selected ", isel, " distance ", dmax[isel])

    return LandmarksIdx


def skenter(kernel):
    print("Centering!")
    return KernelCenterer().fit_transform(kernel)

def kpca(kernel,ndim):
    """ Extracts the first ndim principal components in the space
    induced by the reference kernel (Will expect a square matrix) """
    #Centering step
    k = kernel.copy()
    cols=np.mean(k,axis=0);
    rows=np.mean(k,axis=1);
    mean=np.mean(cols);
    for i in range(len(k)):
        k[:,i]-=cols
        k[i,:]-=rows
    k += mean
    # Eigensystem step
    eval, evec = salg.eigh(k ,eigvals=(len(k)-ndim,len(k)-1) )
    eval=np.flipud(eval); evec=np.fliplr(evec)
    pvec = evec.copy()
    for i in range(ndim):
        pvec[:,i] *= 1./np.sqrt(eval[i])

    # Projection step
    return np.dot(k, pvec)

def ookpca(inrefk,inrectk,ndim=2):
    """ Embeds the out of sample points given by input rectangular kernel
    onto the space spanned by the ndim components of the reference points"""

    sqrk = inrefk.copy()
    rectk = inrectk.copy()
    k = skenter(sqrk)
    m = len(rectk)
    n = len(sqrk)
    recc = rectk - np.dot(np.ones((m,n)),sqrk)*1./n - np.dot(rectk,np.ones((n,n)))*1./n + 1./n**2 * np.dot(np.ones((m,n)),sqrk).dot(np.ones((n,n)))

    print("  And now we build a projection ")
    evalo,evec = salg.eigh(k ,eigvals=(len(k)-ndim,len(k)-1) )
    evalo=np.flipud(evalo); evec=np.fliplr(evec)
    pvec = evec.copy()

    for i in range(ndim):
        pvec[:,i] *= 1./np.sqrt(evalo[i])
    print("Done, super quick. ")
    return np.dot(recc,pvec)

def extractsubm(mat,plist):
    """ Quickly extracts a submatrix, for debugging purposes"""
    return mat[np.ix_(plist,plist)]

def hull_distances(data, energy_idx=0, hull=None):
    """ Compute hull distances """

    if hull is None:
        hull = ConvexHull(data)

    # Omit the simplices on the 'top' of the GCH
    hull_facets = np.delete(
        hull.equations,
        np.nonzero(hull.equations[:, energy_idx] > 0.0),
        axis=0
    )

    hull_distance = -1.0 * (
        np.matmul(data, hull_facets[:, 0:-1].T) 
        + hull_facets[:, -1]
    )

    hull_distance_energy = -1.0 * hull_distance / hull_facets[:, energy_idx]
    hull_distance = np.amin(hull_distance, axis=1)
    hull_distance_energy = np.amin(hull_distance_energy, axis=1)

    return hull_distance, hull_distance_energy

class Hull:
    """ At the moment, not being used. Will be starting point for GCH 2.0"""
    def __init__(self,pfile,sigma_e,sigma_c):
        self.pfile   = pfile
        self.sigma_e = sigma_e
        self.sigma_c = sigma_c
    def update(self,n_pfile,n_sigma_e,n_sigma_c):
        self.pfile   = n_pfile
        self.sigma_e = n_sigma_e
        self.sigma_c = n_sigma_c
