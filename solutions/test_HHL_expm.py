#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/07 

from P2 import *
from tiny_q import *


theta = 2 * pi / (1 << 2)   # = 2*pi / (1<<n_qft) = pi/2
u = spl.expm(1j* A * theta)

eigen_A(A)
eigen_A(u)

eigen_A(np.linalg.matrix_power(u, 2**0))
eigen_A(np.linalg.matrix_power(u, 2**1))
eigen_A(np.linalg.matrix_power(u, 2**2))
