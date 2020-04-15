import numpy as np
import csv
import matplotlib.pyplot as plt

file_path = "./results/origin_random/1mydata_bak.csv"
interval = 10
# plot reward figure
result = []
plt.figure()
plt.xlabel("episodes")
plt.ylabel("reward")
with open(file_path, "r") as f:
    reader = csv.reader(f)
    tmp = []
    for row in reader:
        tmp.append(eval(row[1]) + 600)
        if len(tmp) == interval:
            result.append(np.mean(tmp))
            tmp = []
    plt.plot(np.array(result), 'r')
    plt.show()

# plot win rate figure
result = []
plt.figure()
plt.xlabel("episodes")
plt.ylabel("win rate")
with open(file_path, "r") as f:
    reader = csv.reader(f)
    tmp = []
    for row in reader:
        tmp.append(eval(row[5]) * 10 +5)
        if len(tmp) == interval:
            result.append(np.mean(tmp))
            tmp = []
    plt.plot(np.array(result), 'r')
    plt.show()
