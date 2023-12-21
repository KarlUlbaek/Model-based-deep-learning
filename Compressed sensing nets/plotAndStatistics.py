import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage
import numpy as np

plt.style.use("ggplot")
@torch.no_grad()
def PSNR(MSE, max_=torch.tensor(255.0)):
   MSE = torch.tensor(MSE).mean()
   return (20 * torch.log10(max_) - 10 * torch.log10(MSE)).item()

@torch.no_grad()
def PSNR2(pred, target):
   pred = ((pred + 1)/2) *255.0
   target = ((target + 1)/2) *255.0
   MSE = torch.nn.functional.mse_loss(pred, target)
   return (20 * torch.log10(torch.tensor(255.0)) - 10*torch.log10(MSE)).item()

@torch.no_grad()
def plotLoss(lossToPlot, lrs, lenTrainDataLoader):
   #1AllArray = np.asarray(l1All)#torch.stack(l1All).cpu().numpy()
   #l2AllArray = np.asarray(l2All)#torch.stack(l2All).cpu().numpy()
   lrsArray = np.array(lrs)
   lrsArray = lrsArray / ((lrsArray.max()) * 10)

   for key, value in lossToPlot.items():
      if key != "loss":
         plt.plot(value, label=key)

   plt.plot(lossToPlot["loss"], label="combined loss")

   plt.plot(np.arange(len(lrs)) * lenTrainDataLoader, lrsArray, label="lr")
   plt.legend()
   plt.tight_layout()
   plt.ylim(-0.01, 0.1)
   plt.show()

@torch.no_grad()
def plotPSNR(PSNR):
   PSNRtrain = PSNR["train"]
   PSNRtest = PSNR["test"]
   plt.plot(np.arange(1, len(PSNRtrain)+1), PSNRtrain, label=f"train {PSNRtrain[-1]:.1f}")
   xRange = (np.arange(1, len(PSNRtest)+1)*(int(len(PSNRtrain)) / len(PSNRtest)))
   plt.plot(xRange, PSNRtest, label=f"test {PSNRtest[-1]:.1f}")
   plt.xlabel("epochs")
   plt.title("PSNR")
   plt.legend()
   plt.tight_layout()
   plt.show()


@torch.no_grad()
def plotRecon(x, x_recon, numImgsToPLot=3):
   x = ((x + 1.0) / 2.0).cpu()[:numImgsToPLot, ...]
   x_recon = ((x_recon + 1.0) / 2.0).cpu()[:numImgsToPLot, ...]
   x_recon = torch.clip(x_recon, 0.0, 1.0)

   toPILImage = ToPILImage()
   fig, ax = plt.subplots(2, numImgsToPLot)
   for i in range(numImgsToPLot):
      img = toPILImage(x[i, ...])
      ax[0, i].imshow(img, cmap="gray")
      ax[0, i].axis("off")

      imgRecon = toPILImage(x_recon[i, ...])
      ax[1, i].imshow(imgRecon, cmap="gray")
      ax[1, i].axis("off")
   fig.suptitle(f"Original images (top) versus reconstructed (bottom)")
   plt.tight_layout()
   plt.show()