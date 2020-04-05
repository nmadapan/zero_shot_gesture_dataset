import numpy as np

a = np.random.randint(0, 10, (2, 3))
b = np.random.randint(0, 10, (2, 3))
print(a)
print(b)
print(np.maximum(a, b))
