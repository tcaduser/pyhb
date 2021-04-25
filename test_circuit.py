import math
import numpy as np
from scipy import linalg

class circuit:
    def __init__(self):
        # diode parameters
        self.IS = 1e-16
        self.VT = 0.0259
        # add diffusion capacitance later
        self.TF=1e-7
        # series resistance
        self.RS = 1e3
        # bias
        self.V  = 0
        # solution
        self.solution = np.array([0.0]*3)

    def set_bias(self, v):
        print("set bias %g" % v)
        self.V = v

    def set_solution(self, sol):
        print("set solution %s" % str(sol))
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

        #https://en.wikipedia.org/wiki/Diffusion_capacitance
        qdiode = idiode * self.TF
        qdiode_V2 = idiode_V2 * self.TF

        # series resistance
        grs = 1./self.RS
        irs = (V1 - V2)*grs

        # voltage source
        ibias = V1 - self.V
        dibias_dv1 = 1.
        dI_dI = 1.0

        ivec = np.array([
          ibias,
          +irs + I,
          -irs + idiode
        ])

        gmat = np.array([
          [0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.],
        ])

        gmat[0][1] = dibias_dv1
        gmat[1][0] = +dI_dI
        gmat[1][1] = grs
        gmat[1][2] = -grs
        gmat[2][1] = -grs
        gmat[2][2] = +grs + idiode_V2

        qvec = np.array([
          0.0,
          0.0,
          qdiode
        ])

        cmat = np.array([
          [0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.],
        ])
        cmat[2][2] = qdiode_V2

        return {
          'gmat' : gmat,
          'ivec' : ivec,
          'cmat' : cmat,
          'qvec' : qvec,
        }

    def dc_solve(self):
        for n in range(5):
            data  = self.load_circuit()
            update = -np.dot(linalg.inv(data['gmat']),data['ivec'])
            print(update)
            newsol = self.get_solution() + update
            self.set_solution(newsol)
            #print(update.shape)
            #print(f.shape)

