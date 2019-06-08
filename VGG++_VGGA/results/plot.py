import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_info_from_log(filename="VGG++.out"):
    mode = 0
    mode_str = ["train","val","test"]
    losses = {"train":[], "val":[], "test":[]}
    accs = {"train":[], "val":[], "test":[]}
    with open(filename,"r",encoding="utf-8") as f:
        for line in f:
            if "Loss:" in line:
                loss = float(line[43:48])
                correct = int(line[line.rfind("(")+1:line.find("/")])
                total = int(line[line.find("/")+1:line.rfind(")")])
                acc = float(correct) / total
                part = mode_str[mode]
                #print(f"epoch {epoch}, {part}, loss:{loss}, acc:{acc}")
                losses[part].append(loss)
                accs[part].append(acc)
                mode += 1
                if mode >= 3:
                    mode = 0
    epoches = list(range(len(accs["test"])))
    return epoches, losses, accs

def plot_lr_testacc():
    offset, end = 10, 250
    epoches1, losses1, accs1 = get_info_from_log("VGG++_lr0.008.out")
    best1 = np.max(accs1["test"])
    epoches2, losses2, accs2 = get_info_from_log("VGG++.out")  # lr=0.01
    best2 = np.max(accs2["test"])
    epoches3, losses3, accs3 = get_info_from_log("VGG++_lr0.011.out")
    best3 = np.max(accs3["test"])
    plt.plot(epoches1[offset:end], accs1["test"][offset:end],"r",label="lr=0.008, best_acc=%.4f" % best1)
    plt.plot(epoches2[offset:end], accs2["test"][offset:end],"g",label="lr=0.01, best_acc=%.4f" % best2)
    plt.plot(epoches3[offset:end], accs3["test"][offset:end],"b",label="lr=0.011, best_acc=%.4f" % best3)
    plt.xlabel("epoch")
    plt.ylabel("test_acc")
    plt.legend()
    plt.savefig("lr_acc_plot.jpg")
    plt.show()

def plot_decay_testacc():
    offset, end = 10, 250
    epoches1, losses1, accs1 = get_info_from_log("VGG++_decay0.0001.out")
    best1 = np.max(accs1["test"])
    epoches2, losses2, accs2 = get_info_from_log("VGG++.out")  # decay=0.0005
    best2 = np.max(accs2["test"])
    epoches3, losses3, accs3 = get_info_from_log("VGG++_decay0.001.out")
    best3 = np.max(accs3["test"])
    plt.plot(epoches1[offset:end], accs1["test"][offset:end],"r",label="decay=0.0001, best_acc=%.4f" % best1)
    plt.plot(epoches2[offset:end], accs2["test"][offset:end],"g",label="decay=0.0005, best_acc=%.4f" % best2)
    plt.plot(epoches3[offset:end], accs3["test"][offset:end],"b",label="decay=0.001, best_acc=%.4f" % best3)
    plt.xlabel("epoch")
    plt.ylabel("test_acc")
    plt.legend()
    plt.savefig("decay_acc_plot.jpg")
    plt.show()

plot_lr_testacc()
plot_decay_testacc()
