
from torch.utils.data import Dataset, DataLoader
import os

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as tvt2
from PIL import Image, ImageOps


class CustomImageDataset(Dataset):
   def __init__(self, path, l=1, d="cuda", crop=(96, 96), transform=None, train=True):
      self.train = train
      self.transform = transform
      self.randomCrop = tvt2.RandomCrop(crop)
      self.centerCrop = tvt2.CenterCrop(crop)
      ToTensor = tvt2.Compose([tvt2.ToImageTensor(), tvt2.ConvertImageDtype()])
      self.images = []
      for file in os.listdir(path):
         image = Image.open(os.path.join(path, file))
         if l == 1:
            image = ImageOps.grayscale(image)

         image = (ToTensor(image) * 2.0 - 1.0).to(d)
         if not self.train:
            image = self.centerCrop(image)
         self.images.append(image)
      if self.train:
         flippedH = [tvt2.RandomHorizontalFlip(p=1.0)(img) for img in self.images]
         flippedV = [tvt2.RandomVerticalFlip(p=1.0)(img) for img in self.images]
         flippedHV = [tvt2.RandomVerticalFlip(p=1.0)(img) for img in flippedH]
         self.images = self.images + flippedH + flippedV + flippedHV

   def __len__(self):
      return len(self.images)

   def __getitem__(self, idx):
      image = self.images[idx]
      if self.train:
         image = self.randomCrop(image)

      if self.transform is not None:
         image = self.transform(image)

      return image

if __name__ == "__main__":
   import tqdm
   data = CustomImageDataset(path=r"C:\Users\Karlu\Desktop\11\Model Based Deep Learning\CSnet\images\train")
   dataLoader = DataLoader(dataset=data, batch_size=10, shuffle=True)
   for batch in tqdm.tqdm(dataLoader):
      pass