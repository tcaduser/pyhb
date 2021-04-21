
class hbconfig:
    def __init__(self):
        pass

    # number of harmonics should be base 2 for most efficiency
    # frequencies
    # start with 1D 
    def set_dimensions(number_harmonics):
        '''
        number_harmonics: Number of harmonics to consider
        '''
        self._number_harmonics = number_harmonics


if __name__ == '__main__':
    hb = hbplan()
    print("done")