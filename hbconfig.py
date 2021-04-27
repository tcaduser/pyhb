import numpy as np
import scipy.fft
from scipy import linalg
from scipy import sparse
import scipy.sparse.linalg
import math

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

ddt_scale = -1j
idt_scale = -ddt_scale
two_pi = 2.0 * math.pi

#
# TODO: prevent needless copying by using half ac magnitude
#
# def ss_to_ds_spectrum(avec, harmonic_factor):
#     if harmonic_factor != 1.0:
#         dvec = avec.copy()
#         dvec[1::] *= harmonic_factor
#         return dvec
#     return avec


def real_ifft(dvec):
    #assume it is odd
    if dvec.ndim == 2 and dvec.shape[1] != 1:
        dlen = dvec.shape[1]
    elif dvec.ndim == 1 or dvec.shape[1] == 1:
        dlen = dvec.shape[0]
    else:
        raise RuntimeError("Cannot process " + str(dvec.shape))
    rdata = scipy.fft.irfft(dvec, n=(2*dlen-1), norm='forward')
    return rdata

def real_to_complex_fft(dvec):
    if dvec.ndim == 2 and dvec.shape[1] != 1:
        dlen = dvec.shape[1]
    elif dvec.ndim == 1 or dvec.shape[1] == 1:
        dlen = dvec.shape[0]
    else:
        raise RuntimeError("Cannot process " + str(dvec.shape))
    # note for even number of samples, the last element
    # is purely real, and twice the magnitude
    if not dlen // 2:
        raise RuntimeError("must be careful with high frequency component")
    cdata = scipy.fft.rfft(dvec, norm='forward')
    return cdata


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

    def get_hb_solution(self):
        return self._hb_solution

    def set_hb_solution_update(self, upd):
        nupd = upd.reshape((self._number_rows, self._real_frequency_vec_len))
        nupd[:,0] = np.real(nupd[:,0])
        rerr = linalg.norm(nupd)/linalg.norm(self._hb_solution)
        print("RELATIVE ERROR %g" % rerr)
        self._hb_solution += nupd
        

    def get_hb_solution_time_domain(self):
        hbtd = real_ifft(self._hb_solution)
        return hbtd

    def get_time_bias_vector(self):
        return real_ifft(self._bias_vector)

    def collect_simulation_data(self):
        tbv = self.get_time_bias_vector()
        hbs = self.get_hb_solution_time_domain()
        data = []
        for i, v in enumerate(tbv):
            self._bias_callback(v)
            self._solution_callback(hbs[:,i])
            data.append(self._matrix_rhs_callback())
        #print(data)
        self._time_domain_data = data
        self._preconditioner = None
        return data

    # this is the frequency dependent preconditioner
    # start with dense matrix first
    def get_M_sub_matrix_callback(self):
        data = self._time_domain_data
        gmat = sparse.csc_matrix(data[0]['gmat'].shape, dtype=np.double)
        cmat = sparse.csc_matrix(data[0]['cmat'].shape, dtype=np.double)
        #print("HERE")
        for d in data:
            gmat +=  d['gmat']
            cmat +=  d['cmat']
        scl = 1./float(len(data))
        gmat *= scl
        cmat *= scl
        #print(gmat)
        #print(cmat)
        #raise RuntimeError('TEST')

        def get_M_sub_matrix(wscale):
            Mmat = gmat.astype(np.cdouble)
            tmat = cmat.astype(np.cdouble)
            tmat *= wscale
            Mmat += tmat
            return Mmat

        self._M_sub_matrix_callback = get_M_sub_matrix
        return get_M_sub_matrix;

    def get_omega_scales(self):
        # right now only worry about 1D fft
        ws = [ddt_scale * two_pi * self._fundamental * x for x in range(self._real_frequency_vec_len)]
        #print(ws)
        return ws

    def get_fft_of_td(self, td):
        # assume dimensions are _number_rows, _time_vec_len
        fd = real_to_complex_fft(td)
        return fd


    def apply_omega_scales(self, fd):
        '''
        overwrites copy
        '''
        wscales = self.get_omega_scales()
        for i, v in enumerate(wscales):
            fd[:,i] *= v
        return fd

    # frequency domain RHS
    def get_fd_RHS(self):
        data = self._time_domain_data
        if len(data) != self._time_vec_len:
            raise RuntimeError("UNEXPECTED")
        td_i = np.zeros((self._number_rows, self._time_vec_len), dtype=np.double)
        td_q = np.zeros(td_i.shape, dtype=np.double)
        for i, d in enumerate(data):
            td_i[:,i] = d['ivec']
            td_q[:,i] = d['qvec']

        fd_i = self.get_fft_of_td(td_i)
        fd_q = self.get_fft_of_td(td_q)
        self.apply_omega_scales(fd_q)

        rhs = fd_i
        rhs += fd_q
        rhs = np.reshape(rhs,(self._number_rows*self._real_frequency_vec_len,))
        return rhs

    # assume the solution vector is per frequency per node
    # needs to be converted so that each time-sample jacobian can be readily multiplied by a vector
    def get_td_deltax(self, fdvec):
        '''
        input: solution update vector
        '''
        # npts = (self._number_rows*self._real_frequency_vec_len,)
        fdcopy = fdvec.reshape(self._number_rows, self._real_frequency_vec_len)

        # each column goes to a different jacobian
        #td_deltax = np.zeros((self._number_rows, self._time_vec_len), dtype=np.double)
        #for i,v in enumerate(fdcopy):
        td_deltax = real_ifft(fdcopy)
        return td_deltax

    def apply_jacobian(self, fdvec):
        data = self._time_domain_data

        # dimensions are number of equations by number of time samples
        tdx = self.get_td_deltax(fdvec)

        dg_td = np.zeros((self._number_rows, self._time_vec_len), dtype=np.double)
        dc_td = np.zeros(dg_td.shape, dtype=np.double)

        for i, d in enumerate(data):
            dg_td[:,i] = d['gmat'].dot(tdx[:,i])
            dc_td[:,i] = d['cmat'].dot(tdx[:,i])

        dg_fd = self.get_fft_of_td(dg_td)
        dc_fd = self.get_fft_of_td(dc_td)
        self.apply_omega_scales(dc_fd)

        japplied = dg_fd
        japplied += dc_fd
        #TODO: is this one necessary?
        japplied[:,0] = np.real(japplied[:,0])
        japplied = np.reshape(japplied,(self._number_rows*self._real_frequency_vec_len,))
        return japplied

    def calculate_preconditioner(self):
        cb = self._M_sub_matrix_callback
        wscales = self.get_omega_scales()
        Minvs = [None]*len(wscales)
        for i, w in enumerate(wscales):
            Minvs[i] = sparse.linalg.factorized(cb(w))
        self._preconditioner = Minvs


    def apply_preconditioner(self, fdvec):
        if self._preconditioner == None:
            self.calculate_preconditioner()

        papplied = np.zeros((self._number_rows, self._real_frequency_vec_len), dtype=np.cdouble)
        fdcopy = fdvec.reshape(self._number_rows, self._real_frequency_vec_len)
        for i, p in enumerate(self._preconditioner):
            papplied[:,i] = p(fdcopy[:,i])
        papplied[:,0] = np.real(papplied[:,0])
        papplied = np.reshape(papplied,(self._number_rows*self._real_frequency_vec_len,))
        return papplied

    def get_fd_system_shape(self):
        n = self._number_rows*self._real_frequency_vec_len
        return (n,n)


    def linear_solve(self):
        self.collect_simulation_data()
        self.get_M_sub_matrix_callback()

        fdshape = self.get_fd_system_shape()
        F = -self.get_fd_RHS()

        M_x = lambda x : self.apply_preconditioner(x)
        M = sparse.linalg.LinearOperator(fdshape, M_x)

        J_x = lambda x : self.apply_jacobian(x)
        J = sparse.linalg.LinearOperator(fdshape, J_x)

        #x, exitCode = sparse.linalg.gmres(A=J, b=F, callback_type='pr_norm', callback=lambda x : print(x), tol=1e-20)
        #x, exitCode = sparse.linalg.lgmres(A=J, b=F, M=M, tol=1e-6, maxiter=10)
        x, exitCode = sparse.linalg.gmres(A=J, b=F, M=M, tol=1e-6, restart=10, maxiter=10)
        #x, exitCode = sparse.linalg.gmres(A=J, b=F, M=M, callback_type='pr_norm', callback=lambda x : print(x), tol=1e-20)
        print(x,exitCode)
        print(F)
        #x, exitCode = sparse.linalg.gmres(A=J, b=F, M=M, x0=x, callback_type='pr_norm', callback=lambda x : print(x))
        #print(x,exitCode)
        return x, exitCode

