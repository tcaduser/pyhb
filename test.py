import hbconfig
import test_circuit
import numpy as np


# create circuit
circuit = test_circuit.circuit()

# set dc bias
circuit.set_bias(0.7)

# solve
circuit.dc_solve()

# get solution
sol = circuit.get_solution()
print(sol)
print((sol[1]-sol[2])/circuit.RS)
print(sol[0])

# start with hb config
hb = hbconfig.hbconfig()
nharm = 20
hb.set_harmonics(nharm)
hb.set_bias_callback(lambda x : circuit.set_bias(x))
hb.set_matrix_rhs_callback(lambda : circuit.load_circuit())
hb.set_solution_callback(lambda x : circuit.set_solution(x))
hb.set_fundamental(1e6)

hb.set_number_rows(sol.shape[0])
hb.set_dc_solution(sol)
hb.create_hb_solution()

avec = np.zeros((nharm+1,), dtype=np.cdouble)
avec[0] = 0.7
avec[1] = 0.3
hb.set_bias_vector(avec)

# this is at the beginning of each step
#print(cb(0.0))
# print(cb(0.0))
#print(cb(1.0))

if False:
    hb.get_fd_RHS()
    vec = np.zeros((hb._number_rows*hb._real_frequency_vec_len,))
    hb.apply_jacobian(vec)
    hb.apply_preconditioner()

for i in range(100):
    x, exitCode = hb.linear_solve()
    hb.set_hb_solution_update(x)

#avec[1] = 0.025
#hb.set_bias_vector(avec)
#for i in range(100):
#    x, exitCode = hb.linear_solve()
#    hb.set_hb_solution_update(x)

X = hb.get_hb_solution()
print(X)
Y = X[2,:]
Y = np.concatenate((Y,np.zeros(10)))
x = hbconfig.real_ifft(Y)
import matplotlib.pyplot as plt
plt.plot(x)
#plt.plot(np.flip(x))
plt.show()

#x, exitCode = hb.linear_solve()
#hb.set_hb_solution_update(x)
#x, exitCode = hb.linear_solve()
#hb.set_hb_solution_update(x)
#x, exitCode = hb.linear_solve()
#hb.set_hb_solution_update(x)
#x, exitCode = hb.linear_solve()
#hb.set_hb_solution_update(x)
#x, exitCode = hb.linear_solve()
#hb.set_hb_solution_update(x)
#x, exitCode = hb.linear_solve()
#hb.set_hb_solution_update(x)
#x, exitCode = hb.linear_solve()
#hb.set_hb_solution_update(x)
#x, exitCode = hb.linear_solve()
#hb.set_hb_solution_update(x)
