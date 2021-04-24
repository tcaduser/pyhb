import hbconfig
import test_circuit
from scipy import linalg
import numpy as np


if __name__ == '__main__':
    hb = hbconfig.hbconfig()
    hb.set_harmonics(5)

    circuit = test_circuit.circuit()
    circuit.set_bias(0.5)
    for i in range(5):
        j, f = circuit.load_circuit()
        #print(f)
        #print(linalg.inv(j))
        update = -np.dot(linalg.inv(j),f)
        print(update)
        newsol = circuit.get_solution() + update
        circuit.set_solution(newsol)
        #print(update.shape)
        #print(f.shape)
    sol = circuit.get_solution()
    print(sol)
    print((sol[1]-sol[2])/circuit.RS)
    print(sol[0])

