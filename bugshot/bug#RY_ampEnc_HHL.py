#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/06 

from pyqpanda import *
import numpy as np

# BUG: RYGate is buggy, hence to cause wrong results of Ampltide-Encode and HHL-Algo API!!
# This would be a sweety serious bug that effecting many things in QPanda/Core~

assert RY
assert amplitude_encode
assert HHL_solve_linear_equations

# ↓↓↓ Example ↓↓↓

qvm = CPUQVM()
qvm.init_qvm()
q = qvm.qAlloc()

''' Step 1: RY is buggy, not promising the documentation '''
# Follow the doc https://qpanda-tutorial.readthedocs.io/zh/latest/QGate.html , we have RY = Y(θ):
#    [cos(θ/2), -sin(θ/2)]
#    [sin(θ/2),  cos(θ/2)]
# Hence RY(θ)|0> should give:
#    [cos(θ/2)]
#    [sin(θ/2)]

for _ in range(10):
  # random unit vector b = [b0, b1], where |b| = 1
  b = np.random.uniform(low=-1, high=1, size=[2])
  b /= np.linalg.norm(b)
  assert (np.linalg.norm(b) - 1.0) < 1e-5
  print('b:', b)

  theta = 2 * np.arccos(b[0])     # NOTE: this is not ok, 
  theta = 2 * np.arcsin(b[1])
  b = np.asarray([np.cos(2 * np.arccos(b[0])/2), np.sin(2 * np.arcsin(b[1])/2)])
  print('b:', b)
  
  prog = QProg() << RY(q, theta)
  qvm.directly_run(prog)
  q_b = qvm.get_qstate()
  print('|b>:', np.asarray(q_b).real)

  print()


''' Step 2: Ampltide-Encode is buggy, giving wrong signed answer '''

for _ in range(10):
  # random unit vector b = [b0, b1], where |b| = 1
  b = np.random.uniform(low=-1, high=1, size=[2])
  b /= np.linalg.norm(b)
  assert (np.linalg.norm(b) - 1.0) < 1e-5
  print('b:', b)

  prog = QProg() << amplitude_encode(q, b)
  qvm.directly_run(prog)
  q_b = qvm.get_qstate()
  print('|b>:', np.asarray(q_b).real)

  print()

''' Step 3: HHL-Algo is buggy, giving wrong signed answer '''

