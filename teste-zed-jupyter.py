# %% Teste do kernel WDBC

import sys

print("Python executável:")
print(sys.executable)

print("\nVersão:")
print(sys.version)
# %% Teste das bibliotecas

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("pandas:", pd.__version__)
print("numpy:", np.__version__)


# %% Teste de gráfico

plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Zed + Jupyter + Conda funcionando")
plt.show()
