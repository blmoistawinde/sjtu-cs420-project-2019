import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# extract structured result from log
results = {"train_loss":[],"val_loss":[],"val_acc":[],"test_acc":0.6698}

with open("Resnet18_128.log") as f:
    for line in f:
        if "train_avg_loss: " in line:
            loss = float(line.strip()[-6:])
            results["train_loss"].append(loss)
        elif "val_avg_loss" in line:
            loss = float(line.strip()[14:14+6])
            acc = float(line.strip()[27:27+6])
            results["val_loss"].append(loss)
            results["val_acc"].append(acc)
print(results)
with open("results.json","w") as f:
    json.dump(results,f)

# plot
x = np.arange(len(results["train_loss"]))
plt.plot(x,results["train_loss"],label="train_loss")
plt.plot(x,results["val_loss"],label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig("loss.jpg")

plt.clf()
plt.plot(x,results["val_acc"],label="val_acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()
plt.savefig("acc.jpg")
