import numpy as np
import matplotlib.pyplot as plt
from scaleinvariance.analysis import structure_function_hurst, haar_fluctuation_hurst, spectral_hurst

data = np.cumsum(np.random.randn(2048))

try:
    h_sf = structure_function_hurst(data)
    print(f"Structure function: H = {h_sf:.3f}")
except NotImplementedError:
    print("Structure function: Not yet implemented")

try:
    h_haar = haar_fluctuation_hurst(data)
    print(f"Haar fluctuation: H = {h_haar:.3f}")
except NotImplementedError:
    print("Haar fluctuation: Not yet implemented")

try:
    h_spec = spectral_hurst(data)
    print(f"Spectral method: H = {h_spec:.3f}")
except NotImplementedError:
    print("Spectral method: Not yet implemented")

plt.plot(data)
plt.show()