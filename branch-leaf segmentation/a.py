import random

import numpy as np

data = np.random.randn(10,6)
print(data)
np.random.shuffle(data)
print(data)