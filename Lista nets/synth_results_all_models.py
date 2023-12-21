from custom_data_loader import custom_data_loader
from All_lista_models import *
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from synthetic import sample_synth



def train(train_loader, model, epochs=5, lr = 0.001):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    mses, sparsity, alpha = 0.0, 0.0, 0.0
    prog = tqdm.tqdm(range(epochs))
    mses_to_plot = []
    unstable = False
    for epoch in prog:
        prog.set_postfix({"Name": model.__class__.__name__,"MSE": mses, "sparsity": sparsity, "avg alpha": alpha})
        mse, sparse_code_model_preds = [], []
        for batch_num, (x_batch, y_batch) in enumerate(train_loader):
            sparse_code_model_pred = model(x_batch)
            loss = loss_fn(sparse_code_model_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mse.append(loss.detach())
            sparse_code_model_preds.append(sparse_code_model_pred.detach())
        #print([model.W1[i].weight.grad.mean().item() for i in range(5)])
        #print([model.W1[i].weight.grad.max().item() for i in range(5)])
            if loss > 0.1 and epoch > 0:
                unstable = True
                break

        mses_to_plot.append(torch.stack(mse).cpu().numpy())
        mses = torch.stack(mse).mean().item()
        sparsity = torch.mean((torch.concat(sparse_code_model_preds) == 0).to(torch.float)).item()
        thresh_holds = []
        for module in model.modules():
            if isinstance(module, soft_thresh_module_ss):
                thresh_holds.append(torch.mean(module.threshold.data).item())
        alpha = [str(thresh_hold)[:5] for thresh_hold in thresh_holds]

        if unstable:
            print("unstable and broken")
            mses_to_plot = mses_to_plot[:-1]
            break

    mses_to_plot = np.concatenate(mses_to_plot).astype(float)
    return mses_to_plot, unstable, sparsity

if __name__ == "__main__":
    torch.set_default_dtype(torch.float)
    torch.set_default_device('cuda')
    res = 32
    #split = "train"
    #name = "sampled_data"

    train_loader = custom_data_loader()#split_name=split, name=name)

    n_samples, non_zero_p, noise = 100000, 0.05, 0.01
    W_synth, data_synth, code_synth = sample_synth(res ** 2, n_samples, non_zero_p, noise)
    W_synth = W_synth
    train_loader.x = data_synth
    train_loader.y = code_synth
    train_loader.n = n_samples

    epochs = 10
    lr = 0.001
    batch_size = 1000

    ss_size = 0.01 #or False/None
    learnable_alpha = True
    bias = True
    alpha_init = 0.5
    T = 5

    #train_loader.set_batch_size(batch_size)

    # models = [
    # lista_paper(n=res ** 2, alpha=alpha_init, T=T, learnable_alpha=learnable_alpha, bias=bias, ss_size=ss_size),
    # baseline(n=res ** 2, T=T, alpha=0.001, learnable_alpha=learnable_alpha, bias=bias, ss_size=ss_size),
    # baseline_with_skipcon(n=res ** 2, T=T, alpha=0.01, learnable_alpha=learnable_alpha, bias=bias, ss_size=ss_size),
    # lista_w0_sk(n=res ** 2, alpha=alpha_init, T=T, learnable_alpha=learnable_alpha, bias=bias, ss_size=ss_size),
    # lista_wk_sk(n=res ** 2, alpha=alpha_init, T=T, learnable_alpha=learnable_alpha, bias=bias, ss_size=ss_size),
    # lista_cp_wk(A=W_synth, n=res ** 2, alpha=alpha_init, T=T, learnable_alpha=learnable_alpha, bias=bias, ss_size=ss_size),
    # lista_cp_w0(A=W_synth, n=res ** 2, alpha=alpha_init, T=T, learnable_alpha=learnable_alpha, bias=bias, ss_size=ss_size)
    # ]

    params = [(5, 0.01, 25, 5), (5, False, 2000, 100), (5, 0.01, 2000, 100), (5, 0.01, 100, 10), (2, 0.01, 100, 10), (10, 0.01, 100, 10)]

    plt.style.use("ggplot")
    for (T, ss_size, batch_size, epochs) in params:
        data = []
        title = f"T={T},  ss_size={ss_size},  batch_size={batch_size},  epochs={epochs}"
        train_loader.set_batch_size(batch_size)
        print("running: ", title)

        models = [
            lista_paper(n=res ** 2, alpha=alpha_init, T=T, learnable_alpha=learnable_alpha, bias=bias, ss_size=ss_size),
            baseline(n=res ** 2, T=T, alpha=0.001, learnable_alpha=learnable_alpha, bias=bias, ss_size=ss_size),
            baseline_with_skipcon(n=res ** 2, T=T, alpha=0.0001, learnable_alpha=learnable_alpha, bias=bias,ss_size=ss_size),
            lista_w0_sk(n=res ** 2, alpha=alpha_init, T=T, learnable_alpha=learnable_alpha, bias=bias, ss_size=ss_size),
            lista_wk_sk(n=res ** 2, alpha=alpha_init, T=T, learnable_alpha=learnable_alpha, bias=bias, ss_size=ss_size),
            lista_cp_wk(A=W_synth, n=res ** 2, alpha=alpha_init, T=T, learnable_alpha=learnable_alpha, bias=bias, ss_size=ss_size),
            lista_cp_w0(A=W_synth, n=res ** 2, alpha=alpha_init, T=T, learnable_alpha=learnable_alpha, bias=bias, ss_size=ss_size)
        ]
        min_loss = 1000
        for model in models:
            #print("running:", model.__class__.__name__, "\n")
            mses_to_plot, unstable, sparsity = train(train_loader, model, epochs, lr)
            mses_to_plot = ewma(mses_to_plot, 10)
            name = model.__class__.__name__
            if unstable:
                name += " (stopped)"
            name += "_" + str(sparsity)[:4]
            data.append((name, mses_to_plot))

            plt.figure(figsize=(10, 6))
            for name, d in data:
                plt.plot(d, label=name)
                plt.legend()

            if min_loss > mses_to_plot[-1]:
                min_loss = mses_to_plot[-1]
            plt.title(title + ",  min loss=" + str(min_loss))
            plt.xlabel("steps")
            plt.ylabel("log loss")
            #plt.ylim(-0.01, 0.055)
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig("temp" + title+".png", dpi=150)
            plt.show()
        #plt.show()

        # plt.legend()
        # plt.draw()
        # plt.pause(0.001)

