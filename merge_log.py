import pandas as pd
import numpy as np



df = pd.read_csv("aw_results.txt",header=None,sep=",|:",)
acc = df.iloc[:,3].values.tolist()
results = []
tmp= []
for i,a in enumerate(acc):
    tmp.append(a)
    if (i+1) % 150 == 0:
        results.append(tmp)
        tmp = []

print(len(results))

results = np.array(results[:-1])
max_acc = np.max(results,axis=1)

mean = []
std = []
tmp = []

for i,m in enumerate(max_acc):
    tmp.append(m)
    if (i+1) % 3 == 0:
        mean.append(np.mean(tmp))
        std.append(np.std(tmp))
        tmp = []

print(np.max(mean))
print(np.argmax(mean))
print(std[np.argmax(mean)])
print(mean[np.argmax(mean)])
print(mean)