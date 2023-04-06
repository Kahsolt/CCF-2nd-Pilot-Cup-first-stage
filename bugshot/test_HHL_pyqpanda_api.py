#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/05 

from pyqpanda import *
import numpy as np

r2 = np.sqrt(2)
A = np.asarray([          # eigval: 1.34463698, -1.05174376
  [   1,     1],          # eigvec: [0.9454285, 0.32582963], [-0.43812257, 0.89891524]
  [1/r2, -1/r2],
]).astype(np.complex128)
b = np.asarray([
    1/2, 
  -1/r2,
])

A_ex = np.zeros([4, 4]).astype(np.complex128)
A_ex[:2, 2:] = A
A_ex[2:, :2] = A.conj().T
b_ex = np.zeros([4])
b_ex[:2] = b

x = HHL_solve_linear_equations(A_ex.flatten(), b_ex, precision_cnt=2)
print('>> solution x:', np.abs(np.asarray(x[2:])), '>> |x|:', np.linalg.norm(np.asarray(x[2:])))
z = np.linalg.solve(A, b)
print('>> correct:', z)

print()

# solve random linear equations
for _ in range(10):
  A = np.random.uniform(size=[2, 2], low=-1.0, high=1.0).astype(np.complex128)
  b = np.random.uniform(size=[2],    low=-1.0, high=1.0)

  A = np.asarray([
    [1,  1],
    [1, -1],
  ]) / 2
  try:
    z = np.linalg.solve(A, b)
  except np.linalg.LinAlgError:   # ignore bad rand case if singular
    continue

  x = HHL_solve_linear_equations(A.flatten(), b, precision_cnt=2)
  print('>> solution x:', np.abs(np.asarray(x)), '>> |x|:', np.linalg.norm(np.asarray(x)))
  print('>> correct:', z)
