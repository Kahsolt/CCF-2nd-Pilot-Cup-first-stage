#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/05 

from P2 import *


def go(A, b, n_prec=0, prt=True):
  x = HHL_solve_linear_equations(A.flatten(), b, precision_cnt=n_prec)
  if prt: print('>> solution x:', np.abs(np.asarray(x)), ', |x|:', np.linalg.norm(np.asarray(x)))
  z = np.linalg.solve(A, b)
  if prt: print('>> correct:', z)

  return x


def benchmark(kind:str, eps=1e-2):
  v_errors = 0.0    # error of value L2
  n_errors = 0.0    # error up to a normalization

  for _ in range(100):
    if kind == 'random':
      A_ = np.random.uniform(size=[2, 2], low=-1.0, high=1.0)
      b_ = np.random.uniform(size=[2],    low=-1.0, high=1.0)
    elif kind == 'target':     # around target case
      A_ = A + np.random.uniform(size=[2, 2], low=-1.0, high=1.0) * eps
      b_ = b + np.random.uniform(size=[2],    low=-1.0, high=1.0) * eps
    else: raise ValueError

    A_ = A_ @ A_.conj().T
    b_ /= np.linalg.norm(b_)
    try:
      z = np.linalg.solve(A_, b_)
      z_n = z / np.linalg.norm(z)
    except np.linalg.LinAlgError:
      continue

    x = go(A_, b_, prt=False)
    v_errors += np.linalg.norm(np.abs(z - np.asarray(x)))
    x_n = x / np.linalg.norm(x)
    n_errors += np.linalg.norm(np.abs(z_n - x_n))
  
  print(f'{v_errors} / {n_errors}') 


# solve essay/target equation
go(A12, b1)

exit()

# solve random equations
for _ in range(10):
  A = np.random.uniform(size=[2, 2], low=-1.0, high=1.0).astype(np.complex128)
  A = A @ A.conj().T
  b = np.random.uniform(size=[2],    low=-1.0, high=1.0)
  b = b / np.linalg.norm(b)
  go(A, b)

# benchmark errors
print('[benchmark random]')
benchmark(kind='random')
print('[benchmark target]')
benchmark(kind='target', eps=1e-2)
