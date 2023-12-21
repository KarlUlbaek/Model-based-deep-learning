#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
import time
import torch
torch.set_num_threads(12)
plt.style.use("ggplot")
import numba as nb

@nb.jit(nb.float64[:](nb.float64[:], nb.float64),nopython=True, nogil=True)
def soft_thresh_jit(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, float(0))

def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, float(0))

def soft_thresh_torch(x, l, zero):
    return torch.sign(x) * torch.maximum(torch.abs(x) - l, zero)

def ista(W, x, alpha = 0.05, nsteps=1000, verbose=True, L = None):
    if L is None: L = np.linalg.norm(W, 2) ** 2
    lr = 1.0/L
    alpha_L = alpha * L
    z = np.zeros(W.shape[1])
    diffs = []
    errors = []
    time0 = time.perf_counter()
    for k in range(1, nsteps):
        e = x - W @ z
        z_bar = z + W.T @ e * lr
        z_bar = soft_thresh(z_bar, alpha_L)

        if verbose:
            time1 = time.perf_counter()
            diffs.append(np.linalg.norm(z_bar - z))
            errors.append(np.linalg.norm(e))
            time0 += (time.perf_counter() - time1)
        z = z_bar.copy()
    final_err = np.linalg.norm(e)
    return z, diffs, errors, final_err, time.perf_counter()-time0

# create IST solution
# def ista_njit(W, x, alpha = 0.05, nsteps=1000, verbose=True):
#     L = np.linalg.norm(W, 2) ** 2
#     lr = 1.0/L
#     alpha_L = alpha * L
#     z = np.zeros(W.shape[1])
#     diffs = []
#     errors = []
#     time0 = time.perf_counter()
#     for k in range(1, nsteps):
#         e = x - W @ z
#         z_bar = z + W.T @ e * lr
#         z_bar = soft_thresh_jit(z_bar, alpha_L)
#
#         if verbose:
#             time1 = time.perf_counter()
#             diffs.append(np.linalg.norm(z_bar - z))
#             errors.append(np.linalg.norm(e))
#             time0 += (time.perf_counter() - time1)
#         z = z_bar.copy()
#     final_err = np.linalg.norm(e)
#     return z, diffs, errors, final_err, time.perf_counter()-time0


def fista(W, x, alpha=0.05, nsteps=1000, verbose=True, L =None):
    if L is None: L = np.linalg.norm(W, 2) ** 2
    #print("fista L", L)
    lr = 1 / L
    alpha_L = alpha / L
    z = np.zeros(W.shape[1])
    t = 1
    diffs = []
    errors = []
    time0 = time.perf_counter()
    for k in range(nsteps):
        t_old = t
        e = x - W @ z
        z_bar = z + W.T @ e * lr
        z_bar = soft_thresh(z_bar, alpha_L)
        t = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        z_bar = z + ((t_old - 1) / t) * (z_bar - z)

        if verbose:
            time1 = time.perf_counter()
            diffs.append(np.linalg.norm(z_bar - z))
            errors.append(np.linalg.norm(e))
            time0 += (time.perf_counter() - time1)
        z = z_bar.copy()
    final_err = np.linalg.norm(e)
    return z, diffs, errors, final_err, time.perf_counter()-time0

def CoD(W, x, alpha=0.05, nsteps=1000, verbose=True):
    w_rows, w_cols = W.shape
    z = np.zeros(w_cols)
    S = np.eye(w_cols) - W.T @ W
    B = W.T @ x
    diffs = []
    errors = []
    time0 = time.perf_counter()

    for k in range(nsteps):
        z_bar = soft_thresh(B, alpha)
        k = np.argmax(np.abs(z-z_bar))
        B = B + S[:,k] * (z_bar[k] - z[k])
        z[k] = z_bar[k]

        if verbose:
            time1 = time.perf_counter()
            diffs.append(np.linalg.norm(z_bar - z))
            errors.append(np.linalg.norm(x - W@z))
            time0 += (time.perf_counter() - time1)

    # take one last step and return that step
    z = soft_thresh(B, alpha)
    final_err = np.linalg.norm(x - W@z)
    return z, diffs, errors, final_err, time.perf_counter()-time0, x

def CoD_torch(W, x, alpha=0.05, nsteps=1000, verbose=True):
    w_rows, w_cols = W.shape
    z = torch.zeros(w_cols, dtype=torch.float32)
    S = torch.eye(w_cols) - W.T @ W
    B = W.T @ x
    diffs = []
    errors = []
    time0 = time.perf_counter()
    zero = torch.tensor([0.0], dtype=torch.float32)

    for k in range(nsteps):
        z_bar = soft_thresh_torch(B, alpha, zero)
        k = torch.argmax(torch.abs(z-z_bar))
        B = B + S[:,k] * (z_bar[k] - z[k])
        z[k] = z_bar[k]

        if verbose:
            time1 = time.perf_counter()
            diffs.append(torch.linalg.norm(z_bar - z))
            errors.append(torch.linalg.norm(x - W@z))
            time0 += (time.perf_counter() - time1)

        # take one last step and return that step
    z = soft_thresh_torch(B, alpha, zero)
    z.numpy()
    return z, diffs, errors, time.perf_counter()-time0



if __name__ == "__main__":
    # Signal setup
    N = 1000  # number of observations to make
    l = 2**11  # signal length
    k = 5  # number of nonzero frequency components

    a = [0.3, 1, 0.75, 0.2, 1.2]
    posK = [4, 10, 30, 100, 250]
    np.random.seed(3)

    # Construct the multitone signal
    x = np.zeros(l)
    n = np.arange(l)
    for i in range(k):
        x += a[i]*np.cos((np.pi*(posK[i]-0.5)*n)/l)

    # Construct the sensing matrix
    positions = np.random.randint(l, size=N)
    B = np.zeros((N, l))
    for i in range(N):
        B[i, positions[i]] = 1
    y = B@x
    BF = idct(B, norm='ortho')


    nsteps= 300
    alpha = 0.05 # threshold
    #ista_njit(BF, y, alpha=alpha, nsteps=nsteps)
    ist_sol, ist_diff, ist_err, _, ist_time = ista(BF, y, alpha=alpha, nsteps=nsteps, L=(np.linalg.norm(BF, 2) ** 2))
    #ist_sol_jit, ist_diff_jit, ist_err_jit, _, ist_time_jit = ista_njit(BF, y, alpha=alpha, nsteps=nsteps)
    fist_sol, fist_diff, fist_err, _, fist_time = fista(BF, y, alpha=alpha, nsteps=nsteps, L=(np.linalg.norm(BF, 2) ** 2))
    CoD_sol, CoD_diff, CoD_err, CoD_final_err, CoD_time, _ = CoD(BF, y, alpha=alpha, nsteps=nsteps)
    CoDtorch_sol, CoDtorch_diff, CoDtorch_err, CoDtorch_time = (
        CoD_torch(torch.from_numpy(BF).to(torch.float32),
                  torch.from_numpy(y).to(torch.float32),
                  alpha=alpha, nsteps=nsteps))


    # plot solutions
    fig = plt.figure(figsize=(12,5))
    fig.add_subplot(1, 3, 1)

    range_ = np.arange(l)
    ist_mask = ist_sol > (alpha+1e-6)
    fist_mask = fist_sol > (alpha + 1e-6)
    CoD_mask = CoD_sol > (alpha+1e-6)

    plt.stem(range_[ist_mask], ist_sol[ist_mask], markerfmt='ro', label='IST', basefmt=' ')
    markerline, stemline, baseline = plt.stem(range_[fist_mask], fist_sol[fist_mask], markerfmt='go', label='CoD', basefmt=' ')
    plt.setp(markerline, markersize=10)
    plt.stem(range_[CoD_mask], CoD_sol[CoD_mask], markerfmt='bo', label='fista', basefmt=' ')
    plt.title("non zero coefficients")

    fig.add_subplot(1, 3, 2)
    plt.plot(ist_err, label="ista "+str(ist_time)[:6], color="red")
    #plt.plot(ist_err_jit, label="ista_jit "+str(ist_time_jit)[:6], color="orange")
    plt.plot(fist_err[1:], label="fista "+str(fist_time)[:6], color="blue")
    plt.plot(CoD_err, label="CoD "+str(CoD_time)[:6], color="green")
    plt.plot(CoDtorch_err, label="CoD_torch "+str(CoDtorch_time)[:6], color="orange")
    plt.yscale("log")
    plt.legend()
    plt.title("error")

    fig.add_subplot(1, 3, 3)
    plt.plot(ist_diff, label="ista "+str(ist_time)[:6], color="red")
    plt.plot(fist_diff[1:], label="fista " + str(fist_time)[:6], color="blue")
    plt.plot(CoD_diff, label="CoD "+str(CoD_time)[:6], color="green")
    plt.yscale("log")
    #plt.legend()
    plt.title("l2 change of z (sparse code)")
    plt.tight_layout()
    plt.show()
