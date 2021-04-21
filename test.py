
class hbconfig:
    def __init__(self):
        pass

    # number of harmonics should be base 2 for most efficiency
    # frequencies
    # start with 1D 
    def set_harmonics(number_harmonics):
        '''
        number_harmonics: Number of harmonics to consider
        '''
        self._number_harmonics = number_harmonics
        # so this would be
        # 0, 1, 2, 3, . . . nharm
        # how to make this into an fft number


if __name__ == '__main__':
    hb = hbplan()
    hb.set_harmonics(5)