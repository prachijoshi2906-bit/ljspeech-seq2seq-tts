"""Kernel SVR baseline core (numpy + scikit-learn only).

One epsilon-SVR with a polynomial kernel is trained per output value (mel bin x time step),
as in the CS229 paper (one SVR per time step, multi-output). All SVRs see the same 200
training inputs, so the kernel matrix is computed ONCE and every SVR is fit on it
(kernel="precomputed"). Each fitted SVR is stored as a dense dual-coefficient vector, so
prediction is a single matrix product instead of tens of thousands of SVR.predict calls.
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.svm import SVR
from sklearn.metrics.pairwise import polynomial_kernel


def _kernel(Xa, Xb, gamma, degree):
    return polynomial_kernel(Xa, Xb, degree=degree, gamma=gamma, coef0=1.0)


def _fit_block(K, Y, cols, C, eps):
    A = np.zeros((K.shape[0], len(cols)), dtype=np.float32)
    b = np.zeros(len(cols), dtype=np.float32)
    for j, c in enumerate(cols):
        m = SVR(kernel="precomputed", C=C, epsilon=eps, cache_size=200).fit(K, Y[:, c])
        A[m.support_, j] = m.dual_coef_[0]
        b[j] = m.intercept_[0]
    return A, b


def fit(X, Y, degree=3, C=1.0, eps=0.1, n_jobs=-1, block=500, verbose=True):
    """X: (N, F) features. Y: (N, M) targets (M = n_mels * time steps). Returns a model dict."""
    X = np.asarray(X, dtype=np.float64)
    gamma = 1.0 / (X.shape[1] * X.var())              # same as sklearn gamma="scale"
    K = _kernel(X, X, gamma, degree)
    blocks = [list(range(s, min(s + block, Y.shape[1]))) for s in range(0, Y.shape[1], block)]
    res = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(_fit_block)(K, Y, cols, C, eps) for cols in blocks)
    A = np.concatenate([r[0] for r in res], axis=1)
    b = np.concatenate([r[1] for r in res])
    return {"X": X, "A": A, "b": b, "gamma": gamma, "degree": degree}


def predict(model, Xnew):
    Xnew = np.asarray(Xnew, dtype=np.float64)
    k = _kernel(Xnew, model["X"], model["gamma"], model["degree"])        # (n_new, N)
    return k @ model["A"] + model["b"]                                     # (n_new, M)


def save(model, path):
    np.savez_compressed(path, **model)


def load(path):
    d = np.load(path)
    return {"X": d["X"], "A": d["A"], "b": d["b"],
            "gamma": float(d["gamma"]), "degree": int(d["degree"])}


def to_frames(y, n_mels, stride, T):
    """Flat prediction (n_mels * T_S,) -> (n_mels, T) frames, linearly interpolated if stride > 1."""
    y = np.asarray(y).reshape(n_mels, -1)                 # (n_mels, T_S)
    if stride == 1:
        out = y[:, :T]
        if out.shape[1] < T:                              # pad by repeating the last frame
            out = np.concatenate([out, np.repeat(out[:, -1:], T - out.shape[1], axis=1)], axis=1)
        return out
    src = np.arange(y.shape[1]) * stride
    dst = np.arange(T)
    return np.stack([np.interp(dst, src, row) for row in y])
