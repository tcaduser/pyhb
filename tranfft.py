

import numpy as np
import scipy.fft


# To ensure understanding about what the transpose of the fft operator means


eye = np.eye(3)
print(eye)
f1 = scipy.fft.fft(eye, axis=0, norm='forward')
print(f1)
f2 = scipy.fft.fft(eye, axis=1, norm='forward')
print(f2)
print(f1-f2)
print(f1.transpose() - f1)
f3 = scipy.fft.ifft(eye, axis=0, norm='forward')
print(f3)
print(f3.dot(f1))

print(f3.transpose() - f3)

