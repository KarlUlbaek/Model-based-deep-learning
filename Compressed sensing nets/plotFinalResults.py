import matplotlib.pyplot as plt
plt.style.use("ggplot")
import json
import numpy as np
import os

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
for file in os.listdir("results"):
   if "0.1" in file:
      with open(os.path.join("results", file), "r") as fp:
         data = json.load(fp)["test"]
      xrange = np.arange(1, len(data) + 1)*20
      ax1.plot(xrange[:25], data[:25])
      ax1.set_title("10% sampling rate")
      ax1.set_ylabel("PSNR")
      ax1.set_xlabel("epochs")

   if "0.4" in file:
      with open(os.path.join("results", file), "r") as fp:
         data = json.load(fp)["test"]
      xrange = np.arange(1, len(data) + 1)*20
      ax2.plot(xrange[:25], data[:25], label=file.split("_")[0])
      ax2.set_title("40% sampling rate")
      ax2.set_ylabel("PSNR")
      ax2.set_xlabel("epochs")

fig.suptitle("test PSNR")
plt.legend()
plt.tight_layout()
plt.savefig("result.png", dpi=150)
plt.show()

