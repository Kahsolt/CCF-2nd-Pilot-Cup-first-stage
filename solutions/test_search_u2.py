#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/08 

from pyqpanda import *
import numpy as np
from numpy import pi

# We need this gate as a U4:
#  [-1+i -1-i]
#  [-1-i -1+i] / 2

tgt = np.asarray([
  [-1+1j, -1-1j],
  [-1-1j, -1+1j],
]) / 2

qvm = CPUQVM()
qvm.init_qvm()
q = qvm.qAlloc()

angles = np.linspace(-2*pi, 2*pi, 100)

for alpha in angles:
  for beta in angles:
    for gamma in angles:
      for delta in angles:
        g = U4(alpha, beta, gamma, delta, q)
        m = np.asarray(g.gate_matrix()).reshape((2, 2))

        if np.allclose(m, tgt, atol=1e-3, rtol=1e-3):
          print(alpha, beta, gamma, delta)
