import math

class circuit:
    def __init__(self):
        # diode parameters
        self.IS = 1e-16
        self.VT = 0.0259
        # add diffusion capacitance later
        # series resistance
        self.RS = 1e3
        # bias
        self.V  = 0
        # solution
        self.solution = [0.0]*4

    def set_bias(self, v):
        self.V = v

    def set_solution(self, sol):
        self.solution = sol

    def get_solution(self):
        return self.solution



    def load_circuit(self):
        # solution[0] = I  # current in voltage source
        # solution[1] = V1 # voltage node attached to voltage source and resistor
        # solution[2] = V2 # voltage node attached to resitor and diode
        # diode
        I  = self.solution[0]
        V1 = self.solution[1]
        V2 = self.solution[2]

        IS = self.IS
        VT = self.VT
        common = IS*(math.exp(V2/VT))
        idiode = common - IS
        idiode_V2 = common/VT

        # series resistance
        grs = 1./self.RS
        irs = (V2 - V1)*grs

        # voltage source
        ibias = V1 - self.V
        dibias_dv1 = 1.
        dI_dI = 1.0

        fvec = [
          ibias,
          irs - I,
          -irs + idiode
        ]

        jmat = [
          [0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.],
        ]

        jmat[0][1] = dibias_dv1
        jmat[1][0] = dI_dI
        jmat[1][1] = grs
        jmat[1][2] = -grs
        jmat[2][1] = -grs
        jmat[2][2] = grs + idiode_V2

        return jmat, fvec

