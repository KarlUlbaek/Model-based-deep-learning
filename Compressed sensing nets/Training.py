import os
import torch
from DataSet import CustomImageDataset
from torch.utils.data import DataLoader
from plotAndStatistics import plotLoss, plotRecon, PSNR, PSNR2, plotPSNR
from torch.optim import AdamW
import tqdm
import json

def train(model,
         path =r"images",
         d = "cuda",
         l = 1,
         crop = (96, 96),
         batchSize = 32,
         plotEveryX = 5,
         epochs = 1000,
         lr = 1e-3,):

   model = model.to(d)

   numParams =sum([param.numel() for param in model.parameters() if param.requires_grad]) / 1e6
   print(f"running {model.__class__.__name__} with {numParams:.2f}m params")
   #print(model)


   pathTrain = os.path.join(path, "train")
   dataSetTrain = CustomImageDataset(path=pathTrain, l=l, d=d, crop=crop, train=True)
   dataLoaderTrain = DataLoader(dataset=dataSetTrain, batch_size=batchSize, shuffle=True)

   pathTest = os.path.join(path, "test")
   dataSetTest = CustomImageDataset(path=pathTest, l=l, d=d, crop=crop, train=False)
   dataLoaderTest = DataLoader(dataset=dataSetTest, batch_size=1000, shuffle=False)

   optim = AdamW(model.parameters(), lr=lr)
   #lrScheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=1, T_mult=3)
   lrScheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.1, total_iters=100)

   lossToPlot = {name:[] for name in model.getLossNames()} # init with correct loss func keys
   PSNR = {"train":[], "test":[]}
   lrs = [lr]
   progBar = tqdm.tqdm(range(epochs), unit="epoch")
   for epoch in progBar:
      model.train()
      for x in dataLoaderTrain:
         pred = model(x)
         loss = model.lossFn(pred, target=x)
         optim.zero_grad()
         loss["loss"].backward()
         optim.step()

         for key, value in loss.items():
            lossToPlot[key].append(value.item())

      lrs.append(lrScheduler.get_last_lr()[0])
      lrScheduler.step()

      with torch.no_grad():
         psnrTrain = PSNR2(pred["yDeep"], x)
         PSNR["train"].append(psnrTrain)
         progBar.set_postfix({"PSNR": psnrTrain})

         if (epoch + 1) % plotEveryX == 0:
            plotLoss(lossToPlot, lrs, len(dataLoaderTrain))

            model.eval()
            xTest = next(iter(dataLoaderTest)) # there is only one batch
            predTest = model(xTest)
            psnrTest = PSNR2(predTest["yDeep"], xTest)
            PSNR["test"].append(psnrTest)
            plotRecon(xTest, predTest["yDeep"], 3)
            plotPSNR(PSNR)

            name = f"{model.__class__.__name__}_e{epoch+1}_r{model.r:.2f}"
            with open(name+".txt", "w") as fp:
               json.dump(PSNR, fp)


if __name__ == "__main__":
   from SalsaNet import SalsaNet
   from CSUNet import CSUNet
   from CSNet import CSNet


   train(model=CSNet(n=7, r=0.1), epochs=500, plotEveryX=20)
   train(model=CSNet(n=7, r=0.4), epochs=500, plotEveryX=20)

   train(model=CSUNet(n=64, r=0.1), epochs=500, plotEveryX=20)
   train(model=CSUNet(n=64, r=0.4), epochs=500, plotEveryX=20)

   train(model=SalsaNet(n=5, r=0.1), epochs=500, plotEveryX=20)
   train(model=SalsaNet(n=5, r=0.4), epochs=500, plotEveryX=20)








