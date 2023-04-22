import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import sys

print("**************************************************************************************************************")
# print("This Code using Apache Spark Version =", sc.version)
print("Python Version =", sys.version)
print("version_info=", sys.version_info)
print("base_prefix=", sys.base_prefix)
print("**************************************************************************************************************")

list1 = [-4, -4, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4]

list2 = [-100, -4, -4, -2, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1,
         1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 100]

datadf1 = pd.DataFrame(list1)
datadf2 = pd.DataFrame(list2)

array_std = datadf1.values
array2_std = datadf2.values

print("\n array_std:\n", array_std)
print("\n np.shape(array_std):", np.shape(array_std))
print("\n np.shape(array_std):", np.shape(array2_std))

print("\n np.mean(array_std):\n", np.mean(array_std))
print("\n np.mean(array_std):\n", np.mean(array2_std))

print("\n np.median(array_std):\n", np.median(array_std))
print("\n np.median(array_std):\n", np.median(array2_std))

print("\n np.std(array_std):\n", np.std(array_std))
print("\n np.std(array_std2):\n", np.std(array2_std))

print("\n stats.mode(array_std):", stats.mode(array_std))
print("\n stats.mode(array_std):", stats.mode(array2_std))

sns.distplot(datadf1, bins=9, hist=True, color='green')
sns.distplot(datadf2, bins=9, hist=True, color='red')
plt.show()
