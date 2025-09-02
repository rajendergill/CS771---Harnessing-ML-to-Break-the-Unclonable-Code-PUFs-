import numpy as np
from scipy.linalg import khatri_rao
from scipy.optimize import nnls
from sklearn.svm import LinearSVC

################################
# Non Editable Region Starting #
################################
def my_fit(X_train, y_train, **kwargs):
################################
#  Non Editable Region Ending  #
################################
    """
    Train a linear model on the ML-PUF data.
    X_train: (n_samples, 8) binary challenges
    y_train: (n_samples,) 0/1 responses
    Returns: w (64,), b (scalar)
    """
    # 1) Map to 64-dim features
    PHI = my_map(X_train)  # (n_samples, 64)

    # 2) Train a LinearSVC
    clf = LinearSVC(
        C=1.0,
        loss="squared_hinge",
        tol=1e-4,
        max_iter=10000,
        fit_intercept=True,
        dual=False
    )
    clf.fit(PHI, y_train)

    # 3) Return weight vector and bias
    w = clf.coef_.ravel()        # (64,)
    b = float(clf.intercept_[0]) # scalar
    return w, b

################################
# Non Editable Region Starting #
################################
def my_map(X):
################################
#  Non Editable Region Ending  #
################################
    """
    Create the 64-dimensional feature vector for each 8-bit challenge.
    X: (n_samples, 8) array of {0,1}
    Returns: feat (n_samples, 64)
    """
    # bit-wise Kroncker (outer) product
    return khatri_rao(X.T, X.T).T  # shape (n_samples, 64)

################################
# Non Editable Region Starting #
################################
def my_decode(w_model):
################################
#  Non Editable Region Ending  #
################################
    """
    Invert a 65-dim arbiter-PUF linear model to 4×64 non-neg del.
    w_model: length-65 array (first-64=w, last=bias)
    Returns: p,q,r,s each length-64
    """
    from scipy.optimize import nnls

    y_full = np.array(w_model, float)  # (65,)
    w = y_full[:-1]                    # (64,)
    b = y_full[-1]                     # scalar

    # Build A (65×256) as in lecture
    A = np.zeros((65, 256), float)
    A[0,0:4] = [0.5, -0.5, 0.5, -0.5]
    for i in range(1,64):
        base = 4*i
        pm   = 4*(i-1)
        A[i, base:base+4] += [0.5, -0.5, 0.5, -0.5]
        A[i, pm:pm+4]     += [0.5, -0.5,-0.5,  0.5]
    # bias row
    A[64, 4*63:4*63+4] = [0.5, -0.5, -0.5, 0.5]

    # solve NNLS
    x_hat, _ = nnls(A, y_full)

    # split delays
    p = x_hat[  0:  64]
    q = x_hat[ 64: 128]
    r = x_hat[128: 192]
    s = x_hat[192: 256]
    return p, q, r, s