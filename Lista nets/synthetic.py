import torch
import numpy as np


def sample_synth(size=32 ** 2, n_samples=1000, non_zero_p=0.2, noise=0.01):
    W_synth = torch.randn(size=(size, size)) * (1 / size)
    W_synth /= torch.linalg.norm(W_synth, dim=0)
    code_synth = torch.bernoulli(torch.ones(n_samples, size) * non_zero_p) * torch.randn(size=(n_samples, size))
    data_synth = (W_synth @ code_synth.T).T + torch.randn(size=(n_samples, size)) * noise

    return W_synth, data_synth, code_synth


if __name__ == "__main__":
    from ista_fista_cod import CoD
    import matplotlib.pyplot as plt
    W_synth, data_synth, code_synth = sample_synth()
    print("mean, std:", data_synth.mean(dim=0)[:3], data_synth.std(dim=0)[:3])

    i = 3
    z, diffs, errors, _ , _, _ = CoD(W=W_synth.numpy(), x=data_synth[i, :].numpy(), verbose=True, nsteps=300)
    plt.plot(errors, label="errors")
    plt.plot(diffs, label="diffs")
    plt.legend()
    print("zero", np.mean(z == 0))
    print(np.linalg.norm(z - code_synth[i, :].numpy()))
    plt.show()