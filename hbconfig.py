import numpy as np
import scipy.fft

#### ALGORITHM
# for each bias condition:
#   collect g, c, i, q
# create preconditioner:
#   average of g
#   average of c
# for each frequency
#   calculate P(j*omega)
#   solve     M = inv(P(omega))
# calculate Omega
#   the time/frequency differentiation operator
#
# Nonlinear iteration:
#   form RHS vector F = \Gamma i + \Omega \Gamma q
#   form dx vector from RHS Vector
#   Linear iteration:
#     Based on Request:
#       Apply M to vector
#       Apply Jacobian to Vector:
#         (a) Apply \Gamma^{-1} to dx vector
#         (b) Apply time domain g to (a)
#         (c) Apply \Gamma to (b)
#         (d) Apply time domain c to (a)
#         (e) Apply \Gamma to to (d)
#         (f) Apply Omega to (e)
#         (g) sum (c) + (f)


def ss_to_ds_spectrum(avec, harmonic_factor):
    if harmonic_factor != 1.0:
        avec[1::] *= harmonic_factor
    dvec = np.concatenate((avec, np.conjugate(np.flip(avec[1::]))))
    return dvec

def real_ifft(dvec):
    rdata = np.real(scipy.fft.ifft(dvec, norm='forward'))
    return rdata

# start with 1D FFT
class hbconfig:
    def __init__(self):
        self._number_harmonics = None
        self._fundamental = None
        self._number_rows = None
        self._initial_guess = None
        # callback to set the bias
        self._bias_callback = None
        # callback to set the time dependent variable
        self._variable_setter = None
        # callback to aquire the time-dependent jacobian and rhs
        self._matrix_rhs_getter = None


    # number of harmonics should be base 2 for most efficiency
    # for now be odd, or zero pad
    # frequencies
    # start with 1D
    def set_harmonics(self, number_harmonics):
        '''
        number_harmonics: Number of harmonics to consider
        '''
        self._number_harmonics = number_harmonics
        # so this would be
        # 0, 1, 2, 3, . . . nharm
        # how to make this into an fft number

    def set_number_rows(self, number_rows):
        self._number_rows = number_rows

    def set_initial_guess(self, initial_guess):
        '''
        set the initial dc solution as the initial guess otherwise it is zero
        '''
        pass

    def set_fundamental(self, fundamental):
        '''
        This is the fundamental frequency for large signal bias
        '''
        self._fundamental = fundamental

    def set_bias_vector(self, biasvector):
        if biasvector.shape[0] != self._real_frequency_vec_len:
            raise RuntimeError('wrong size for bias_vector %d != %d', (biasvector.shape[0], self._real_frequency_vec_len))
        self._bias_vector = biasvector

    # should be able to set multiple sources
    def set_bias_callback(self, cb):
        self._bias_callback = cb

    def set_matrix_rhs_callback(self, cb):
        self._matrix_rhs_callback = cb

    def set_solution_callback(self, cb):
        self._solution_callback = cb

    def set_dc_solution(self, sol):
        self._dc_solution = sol

    def create_hb_solution(self):
        # These are the positive only frequency terms
        # and are what we will actually store and use for iteration
        length = self._number_harmonics + 1
        self._real_frequency_vec_len = length
        self._time_vec_len = 2*length - 1
        self._hb_solution = np.zeros(shape=(self._number_rows, length), dtype=np.cdouble)
        if self._number_rows != self._dc_solution.shape[0]:
            raise RuntimeError('Initial DC solution must be same length _number_rows')
        self._hb_solution[:, 0] = self._dc_solution
        print(self._hb_solution)

    def get_hb_solution_time_domain(self):
        hbtd = np.zeros(shape=(self._number_rows, self._time_vec_len))
        for i in range(self._number_rows):
            hbtd[i,:] = real_ifft(ss_to_ds_spectrum(self._hb_solution[i,:], 0.5))
        return hbtd

    def get_time_bias_vector(self):
        dvec = ss_to_ds_spectrum(self._bias_vector, 0.5)
        return real_ifft(dvec)

    def collect_data(self):
        tbv = self.get_time_bias_vector()
        hbs = self.get_hb_solution_time_domain()
        data = []
        for i, v in enumerate(tbv):
            self._bias_callback(v)
            self._solution_callback(hbs[:,i])
            data.append(self._matrix_rhs_callback())
        print(data)

