import torch
from torch import nn
import torch.nn.functional as F

class SamplingBlock(nn.Module):
   def __init__(self, B=32, l=1, r=0.1):
      super().__init__()

      self.B = B # size of blocks
      self.l = l # num color channels
      self.nB = int(r * l * B**2) # number of filters

      self.sampling = nn.Conv2d(in_channels=l, out_channels=self.nB, kernel_size=(B, B),
                                stride=(B, B), padding=(0,0), bias=False)

   def forward(self, x):

      x_b, x_l, x_h, x_w = x.shape
      assert (x_h % self.B == 0 and
              x_w % self.B == 0)

      y = self.sampling(x)
      return y

class InitReconBlock(nn.Module):
   def __init__(self, B=32, l=1, r=0.1):
      super().__init__()

      self.B = B # size of blocks
      self.l = l # num color channels
      self.nB = int(r * l * B**2) # number of filters/channels
      self.pixelShuffle = nn.PixelShuffle(B)

      self.init_recon = nn.Conv2d(in_channels=self.nB, out_channels=l*B**2, kernel_size=(1,1),
                                  stride=(1,1), padding=(0,0), bias=False)

   def forward(self, x):
      y = self.init_recon(x)
      return self.pixelShuffle(y)


class SoftThreshHold(nn.Module):
   def __init__(self, imgDim = 96, d=32, sameForAllChannels = True, learnable=True):
      super().__init__()
      if sameForAllChannels:
         self.thresholds = torch.nn.Parameter(torch.randn((1, d, 1, 1), dtype=torch.float).abs()/10.0)
      else:
         self.thresholds = torch.nn.Parameter(torch.ones((d, imgDim, imgDim), dtype=torch.float).abs()/100.0)

      if not learnable:
         self.thresholds.requires_grad = False

   def forward(self, x):
      signs = torch.sign(x)
      return signs * (F.relu(x.abs() - self.thresholds))


class ThresholdDenoiseUpdate(nn.Module):
   def __init__(self, l=1, d=32, imgDim = 96, bias=True):
      super().__init__()

      self.c = nn.Conv2d(in_channels=l, out_channels=d, stride=(1, 1),padding=(1,1), kernel_size=(3,3), bias=bias)

      self.h = nn.Sequential(
               nn.Conv2d(in_channels=d, out_channels=d, stride=(1,1),padding=(1,1), kernel_size=(3,3), bias=bias),
               nn.ReLU(),
               nn.Conv2d(in_channels=d, out_channels=d, stride=(1, 1),padding=(1, 1), kernel_size=(3, 3), bias=bias)
               )

      self.b = nn.Conv2d(in_channels=d, out_channels=l, stride=(1, 1), padding=(1,1), kernel_size=(3,3), bias=bias)

      self.hTilde = nn.Sequential(
                    nn.Conv2d(in_channels=d, out_channels=d, stride=(1,1), padding=(1,1), kernel_size=(3,3), bias=bias),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=d, out_channels=d, stride=(1, 1),padding=(1, 1), kernel_size=(3, 3), bias=bias)
                   )

      self.softThreshHold = SoftThreshHold(imgDim=imgDim, d=d, sameForAllChannels=True, learnable=True)


   def forward(self, x, m):
      x_m = x+m
      cOut = self.c(x_m)
      hOut = self.h(cOut)
      out = self.softThreshHold(hOut)
      out = self.hTilde(out)
      out = self.b(out) + x_m

      lSym = torch.nn.functional.mse_loss(self.hTilde(hOut), cOut)
      return out, lSym

class GradientUpdate(nn.Module):
   def __init__(self, samlpingBlock, initReconBlock, imgDim=96, muInitScale = 0.01):
      super().__init__()

      #self.mu = torch.nn.Parameter(torch.rand(imgdim).abs()/muInitScale)
      self.mu = torch.nn.Parameter(torch.tensor(muInitScale))
      self.imgDim = imgDim
      self.samlpingBlock = samlpingBlock
      self.initReconBlock = initReconBlock

   # not actually used
   def forwardAdmmPaper(self, x):
      minus_output = x['minus_output']
      multiple_output = x['multi_output']
      input = x['input']
      mask = self.mask
      rhs = input + self.rho * torch.fft.fft2(minus_output - multiple_output)

      lhs = mask.cuda() + self.rho # this line should be mask + torch.eye(imgDim)*self.rho
      orig_output2 = rhs / lhs
      orig_output3 = torch.fft.ifft2(orig_output2)
      x['re_mid_output'] = orig_output3
      return x


   def forward(self, xk, x0, v, m):
      #import torch_dct as dct
      #muDiag = self.mu*torch.eye(self.imgDim, device=xk.device)
      #lhs = torch.zeros_like(x0)


      lhs = (1/(torch.fft.fft2(self.mu + (xk))))

      rhs = torch.fft.fft2(x0 + self.mu * (v - m))

      xHat = lhs * rhs

      return torch.fft.ifft2(xHat).real

   # not actually used
   def forwardSalsaNet(self, xk, x0, v, m):
      import torch_dct as dct
      muDiag = self.mu*torch.eye(self.imgDim, device=xk.device)
      lhs = muDiag + self.initReconBlock(self.samlpingBlock(xk))

      rhs = x0 + self.mu * (v - m)

      fraction = rhs.inverse() @ lhs

      return fraction


class AuxiliaryUpdate(nn.Module):
   def __init__(self):
      super().__init__()

   def forward(self, m, v, x):
      return m - v + x



class SalsaNet(nn.Module):
   def __init__(self, B=32, l=1, imgDim=96, d=32, n=5, r=0.1, sepConv2d=False):
      super().__init__()
      self.r = r
      self.n = n
      self.samplingBlock = SamplingBlock(B=B, l=l, r=r)
      self.initReconBlock = InitReconBlock(B=B, l=l, r=r)

      self.gradientUpdate = GradientUpdate(self.samplingBlock, self.initReconBlock)
      self.auxiliaryUpdate = AuxiliaryUpdate()

      thresholdDenoiseUpdate = [ThresholdDenoiseUpdate(l=l, d=d, imgDim=imgDim) for _ in range(n)]
      self.thresholdDenoiseUpdate = nn.ParameterList(thresholdDenoiseUpdate)
      # self.thresholdDenoiseUpdate = ThresholdDenoiseUpdate(l=l, d=d, imgDim=imgDim)

   # not actually used
   def forward1(self, xInput):
      m = torch.zeros_like(xInput)
      x0 = self.initReconBlock(self.samplingBlock(xInput))

      xk = torch.clone(x0)
      lSymTot = 0.0
      for k in range(self.n):
         v, lSym = self.thresholdDenoiseUpdate[k](x=xk, m=m)
         xk = self.gradientUpdate(xk=xk, x0=x0, v=v, m=m)
         m = self.auxiliaryUpdate(m=m, v=v, x=xk)
         lSymTot += lSym

      return {"yDeep":xk, "yInit":x0, "lSym":lSymTot}


   def forward(self, xInput):
      m = torch.zeros_like(xInput)
      yInit = self.initReconBlock(self.samplingBlock(xInput))
      lSymTot = 0.0
      v = torch.clone(yInit)
      for k in range(self.n):
         v, lSym = self.thresholdDenoiseUpdate[k](x=v, m=m)
         lSymTot += lSym

      return {"yDeep":v, "yInit":yInit, "lSym":lSymTot}

   def getLossNames(self):
      return ["loss", "lDeep", "lInit", "lSym"]

   def lossFn(self, y, target, wlSym=0.1, wlInit=0.1 ):
      yInit = y["yInit"]
      yDeep = y["yDeep"]
      lSym = y["lSym"]

      lInit = F.mse_loss(yInit, target)
      lDeep = F.mse_loss(yDeep, target)

      loss = lDeep + wlInit*lInit + wlSym*lSym

      return {"loss": loss, "lInit": lInit, "lDeep": lDeep, "lSym": lSym}


if __name__ == "__main__":

   def phi(k, a=-0.4, b=-2):
      return torch.log( 1.0+torch.exp(torch.tensor(a*k + b)) )

   import matplotlib.pyplot as plt
   plt.plot(phi(k=torch.arange(10), a=-0.4, b=-2), label="beta_k")
   plt.plot(phi(k=torch.arange(10), a=-0.2, b=-1), label="mu_k")
   plt.xlabel("k")
   plt.ylabel("$\phi ()$")
   plt.legend()
   plt.show()


   d = "cuda"
   model = SalsaNet().to(d)
   print(model)
   x = torch.randn((50, 1, 96, 96), device=d)

   print("num params:", sum([param.numel() for param in model.parameters() if param.requires_grad]))

   import time

   with torch.no_grad():
      #warmup
      for _ in range(2):
         _ = model(x)
      torch.cuda.synchronize()

      t = time.perf_counter()
      for _ in range(10):
         y_deep = model(x)
      torch.cuda.synchronize()
      print("time:", time.perf_counter()-t)

   print(y_deep.shape)

   print("compression ratio:", x.numel() / y_deep.numel() )