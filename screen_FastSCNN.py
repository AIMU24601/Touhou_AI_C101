import torch
import numpy as np

from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, data_home=".")

x = mnist.data /255
y = mnist.target

import matplotlib.pyplot as plt

print(y[0])
plt.imshow(x[0].reshape(28, 28), cmap="gray")
