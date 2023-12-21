from custom_data_loader import custom_data_loader
from All_lista_models import lista_paper, lista_w0_sk, baseline, soft_thresh_module
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt


device = "cuda"
model = baseline(num_blocks=5, alpha=0.01, learnable_alpha=True)
#model = lista(alpha=0.5, T=3, learnable_alpha=True)
#model = lista_Sk(alpha=0.5, T=5, learnable_alpha=True, bias=False)
model.to(device)
loss_fn = torch.nn.MSELoss()
lr = 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
batch_size = 100
epochs = 100

split = "train"
name = "mnist"
train_set = custom_data_loader(split_name=split, name=name, batch_size=batch_size, device=device)
train_set.x = train_set.x.squeeze()
train_set.x = torch.reshape(train_set.x, train_set.x.shape[:1] + tuple([-1])).to(torch.float32).to(device) - 0.5
train_set.y = torch.load("W_generated_sparse_code_mnist.pt").to(torch.float32).to(device)
W_true = torch.load("W_learned_mnist.pt").to(torch.float32).to(device)

mses, sparsity, alpha = 0.0, 0.0, 0.0
prog = tqdm.tqdm(range(epochs))
for epoch in prog:
    prog.set_postfix({"MSE": mses, "sparsity": sparsity, "avg alpha": alpha})
    mse, sparse_code_model_preds = [], []
    for batch_num, (x_batch, y_batch) in enumerate(train_set):
        sparse_code_model_pred = model(x_batch)
        loss = loss_fn(sparse_code_model_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mse.append(loss.detach())
        sparse_code_model_preds.append(sparse_code_model_pred.detach())

    mses = torch.stack(mse).mean().item()
    sparsity = torch.mean((torch.concat(sparse_code_model_preds) == 0).to(torch.float)).item()
    if epoch % 1 == 0:
        thresh_holds = []
        for module in model.modules():
            if isinstance(module, soft_thresh_module):
                thresh_holds.append(torch.mean(module.threshold.data).item())
        alpha = [str(thresh_hold)[:5] for thresh_hold in thresh_holds]
        #print("batch sparsity:", torch.mean((sparse_code_model_pred == 0).to(torch.float)).item())
        #print("thresholds:", thresh_holds)
        f, ax = plt.subplots(1, 6)
        f.set_figheight(5)
        f.set_figwidth(13)
        plt.tight_layout()
        for i in range(6):
            ax[i].axis("off")
        ax[0].imshow((W_true @ sparse_code_model_pred[0, :]).reshape(28, 28).detach().cpu().numpy(), cmap='gray')
        ax[1].imshow((W_true @ sparse_code_model_pred[1, :]).reshape(28, 28).detach().cpu().numpy(), cmap='gray')
        ax[1].set_title("network rec")
        ax[2].imshow((W_true @ y_batch[0, :]).reshape(28, 28).detach().cpu().numpy(), cmap='gray')
        ax[3].imshow((W_true @ y_batch[1, :]).reshape(28, 28).detach().cpu().numpy(), cmap='gray')
        ax[3].set_title("W rec")
        ax[4].imshow((x_batch[0, :]).reshape(28, 28).cpu().numpy(), cmap='gray')
        ax[5].imshow((x_batch[1, :]).reshape(28, 28).cpu().numpy(), cmap='gray')
        ax[5].set_title("True")
        plt.show()