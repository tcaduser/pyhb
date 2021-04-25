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
hb.set_matrix_rhs_callback(lambda : circuit.load_circuit())
hb.set_solution_callback(lambda x : circuit.set_solution(x))

hb.set_number_rows(sol.shape[0])
hb.set_dc_solution(sol)
hb.create_hb_solution()
avec = np.array([0.5, 0.001, 0, 0, 0, 0], dtype=np.cdouble)
hb.set_bias_vector(avec)
#print(hb.get_time_bias_vector())
hb.collect_simulation_data()
#print(hb.get_hb_solution_time_domain())
cb = hb.get_M_sub_matrix_callback()
print(cb(0.0))
# print(cb(0.0))
print(cb(1.0))



