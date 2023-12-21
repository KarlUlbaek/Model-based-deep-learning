import torch
from torch import nn
from torch.nn.functional import mse_loss
import torch.nn.functional as F
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

""" Full assembly of the parts to form the complete network """
class UNet(nn.Module):
    def __init__(self, inchannels, initChannelNumber=10, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = inchannels
        self.bilinear = bilinear
        c = initChannelNumber

        self.inc = (DoubleConv(inchannels, c))
        self.down1 = (Down(c, c*(2**1)))
        self.down2 = (Down(c*(2**1), c*(2**2)))
        self.down3 = (Down(c*(2**2), c*(2**3)))
        factor = 2 if bilinear else 1
        self.down4 = (Down(c*(2**3), c*(2**4) // factor))
        self.up1 = (Up(c*(2**4), c*(2**3) // factor, bilinear))
        self.up2 = (Up(c*(2**3), c*(2**2) // factor, bilinear))
        self.up3 = (Up(c*(2**2), c*(2**1) // factor, bilinear))
        self.up4 = (Up(c*(2**1), c, bilinear))
        self.outc = (OutConv(c, inchannels))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

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


class CSUNet(nn.Module):
   def __init__(self, B=32, l=1, n=16, r=0.1):
      super().__init__()
      self.r = r
      self.Sampler = SamplingBlock(B=B, l=l, r=r)
      self.InitRecon = InitReconBlock(B=B, l=l, r=r)
      self.unet = UNet(inchannels=l, initChannelNumber=n)

   def forward(self, x):
      samples = self.Sampler(x)
      yInit = self.InitRecon(samples)
      yDeep = self.unet(yInit)# + yInit

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
   model = CSUNet().to(d)
   print(model)
   x = torch.randn((3, 1, 96, 96), device=d)

   print("num params:", sum([param.numel() for param in model.parameters() if param.requires_grad]))

   import time

   with torch.no_grad():
      #warmup
      for _ in range(2):
         _ = model(x)
      if torch.cuda.is_available(): torch.cuda.synchronize()

      t = time.perf_counter()
      for _ in range(10):
         pred = model(x)
      if torch.cuda.is_available(): torch.cuda.synchronize()
      print("time:", time.perf_counter()-t)

   print(pred["yDeep"].shape)

   print("compression ratio:", x.numel() / pred["yDeep"].numel() )