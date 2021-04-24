import hbconfig
import test_circuit
from scipy import linalg
import numpy as np


if __name__ == '__main__':
    hb = hbconfig.hbconfig()
    hb.set_harmonics(5)

    circuit = test_circuit.circuit()
    circuit.set_bias(0.5)
    for i in range(1):
        j, f = circuit.load_circuit()
        #print(f)
        #print(linalg.inv(j))
        update = -np.dot(linalg.inv(j),f)
        print(update)
        #print(update.shape)
        #print(f.shape)

