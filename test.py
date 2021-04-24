import hbconfig
import test_circuit
import numpy as np



# create circuit
circuit = test_circuit.circuit()

# set dc bias
circuit.set_bias(0.5)

# solve
circuit.dc_solve()

# get solution
sol = circuit.get_solution()
print(sol)
print((sol[1]-sol[2])/circuit.RS)
print(sol[0])

# start with hb config
hb = hbconfig.hbconfig()
hb.set_harmonics(5)
hb.set_bias_callback(lambda x : circuit.set_bias(x))
hb.set_matrix_rhs_callback(lambda x : circuit.load_circuit())
hb.set_solution_callback(lambda x : circuit.set_solution(x))
