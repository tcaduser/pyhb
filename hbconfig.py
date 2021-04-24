import numpy as np

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

if __name__ == '__main__':
    hb = hbconfig()
    hb.set_harmonics(5)
