import torch
from torch import nn
from torch.nn.functional import mse_loss



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
      #self.pixelShuffle = nn.PixelShuffle(B)

      self.init_recon = nn.Conv2d(in_channels=self.nB, out_channels=l*B**2, kernel_size=(1,1),
                                  stride=(1,1), padding=(0,0), bias=False)

   def forward(self, x0, x):
      # we need x0 to keep track of the shape and we just calcualete the new vaules inplace in x0
      x_b, x_l, x_h, x_w = x0.shape
      x0Like = torch.empty_like(x0)
      y = self.init_recon(x)

      # block reshape
      #x0 = y.view(x0.shape)
      #x0Like = self.pixelShuffle(y)
      for i in range(int(x_h / self.B)):
         for j in range(int(x_h / self.B)):
            x0Like[:, :, self.B*i:(self.B)*(i+1), self.B*j:(self.B)*(j+1)] = y[:, :, i, j].view(x_b, x_l, self.B, self.B)

      return x0Like

class SepConv2d(nn.Module):
   def __init__(self, d, f, bias=True):
       super().__init__()
       self.depthwise = nn.Conv2d(in_channels=d, out_channels=d, stride=(1,1),
                                   padding=(1,1), kernel_size=(f,f), bias=bias, groups=d)

       self.pointwise = nn.Conv2d(in_channels=d, out_channels=d, stride=(1,1),
                                  kernel_size=1, bias=bias)

   def forward(self, x):
       out = self.depthwise(x)
       out = self.pointwise(out)
       return out

class ResConvBLock(nn.Module):
   def __init__(self, d=64, f=3, bias=True, sepConv2d=False):
      super().__init__()

      self.act = torch.nn.ReLU()
      if not sepConv2d:
         self.conv1 = torch.nn.Conv2d(in_channels=d, out_channels=d, stride=(1,1),
                                      padding=(1,1), kernel_size=(f,f), bias=bias)
         self.conv2 = torch.nn.Conv2d(in_channels=d, out_channels=d, stride=(1,1),
                                      padding=(1,1), kernel_size=(f,f), bias=bias)
      else:
         self.conv1 = SepConv2d(d=d, f=f, bias=bias)
         self.conv2 = SepConv2d(d=d, f=f, bias=bias)

   def forward(self, x):
      y = self.act(self.conv1(x)) + x
      y = self.act(self.conv2(y))

      return y

class DeepReconBlock(nn.Module):
   def __init__(self, d=64, f=3, l=1, n=5, sepConv2d=False):
      super().__init__()

      self.act = torch.nn.ReLU()
      self.FeatureExtractConv = torch.nn.Conv2d(in_channels=l, out_channels=d, stride=(1,1),
                                                padding=(1,1), kernel_size=(f,f), bias=True)

      self.FeatureAggregateConv = torch.nn.Conv2d(in_channels=d, out_channels=l, stride=(1,1),
                                                  padding=(1,1), kernel_size=(f,f), bias=True)

      self.ResConvBlocks = nn.Sequential(*[ResConvBLock(d=d, f=f, sepConv2d=sepConv2d) for _ in range(n)])

   def forward(self, x):
      y = self.act(self.FeatureExtractConv(x))
      y = self.ResConvBlocks(y)
      y = self.FeatureAggregateConv(y)

      return y




class CSNet(nn.Module):
   def __init__(self, B=32, l=1, f=3, d=64, n=5, r=0.1, sepConv2d=False):
      super().__init__()
      self.r = r
      self.Sampler = SamplingBlock(B=B, l=l, r=r)
      self.InitRecon = InitReconBlock(B=B, l=l, r=r)
      self.DeepRecon = DeepReconBlock(d=d, f=f, l=l, n=n, sepConv2d=sepConv2d)

   def forward(self, x):
      samples = self.Sampler(x)
      yInit = self.InitRecon(x, samples)
      yDeep = self.DeepRecon(yInit) + yInit

      return {"yInit":yInit, "yDeep": yDeep}

   def getLossNames(self):
      return ["lInit", "lDeep", "loss"]

   def lossFn(self, y, target, w=0.1):
      yInit = y["yInit"]
      yDeep = y["yDeep"]

      lInit = mse_loss(yInit, target)
      lDeep = mse_loss(yDeep, target)
      loss = lDeep + w*lInit

      return {"lInit": lInit, "lDeep": lDeep, "loss": loss}



if __name__ == "__main__":
   d = "cpu"
   model = CSNet().to(d)
   print(model)
   x = torch.randn((100, 1, 96, 96), device=d)


   # from torch.profiler import profile, record_function, ProfilerActivity
   # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
   #    with record_function("inference"):
   #       for _ in range(8):
   #          model(x)["yDeep"].sum().backward()
   # print("1 epoch")
   # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
   #

   print("num params:", sum([param.numel() for param in model.parameters() if param.requires_grad]))

   import time

   with torch.no_grad():
      #warmup
      for _ in range(2):
         _ = model(x)
      torch.cuda.synchronize()

      t = time.perf_counter()
      for _ in range(10):
         y_deep, y_init = model(x)
      torch.cuda.synchronize()
      print("time:", time.perf_counter()-t)

   print(y_deep.shape)

   #print("compression ratio:", x.numel() / y_deep.numel() )