import hbconfig
import test_circuit
import numpy as np


if __name__ == '__main__':
    hb = hbconfig.hbconfig()
    hb.set_harmonics(5)

    circuit = test_circuit.circuit()
    circuit.set_bias(0.5)
    circuit.dc_solve()
    sol = circuit.get_solution()
    print(sol)
    print((sol[1]-sol[2])/circuit.RS)
    print(sol[0])

