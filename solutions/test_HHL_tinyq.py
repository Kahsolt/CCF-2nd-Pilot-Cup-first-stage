#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/07 

from P2 import *
from random import random, randrange
from tiny_q import *


def HHL(A: np.ndarray, b: np.ndarray, t1=pi, t2=pi/2, r1=pi/8, r2=pi/16) -> State:
  import scipy.linalg as spl

  assert np.allclose(A, A.conj().T), 'A should be a hermitian'
  assert (np.linalg.norm(b) - 1.0) < 1e-5, 'b should be a unit vector'

  ''' enc |b> '''
  enc_b = v('000') @ amplitude_encode(b)

  ''' QPE '''
  u = I @ H @ H @ I

  u_A1 = spl.expm(1j* A * (t2))
  u << (get_I(2) @ Control(Gate(u_A1)))

  swap12 = I @ SWAP @ I
  u << swap12
  u_A2 = spl.expm(1j* A * (t1))
  u << (get_I(2) @ Control(Gate(u_A2)))
  u << swap12

  u << swap12
  u << (get_I(2) @ H @ I)
  u << (I @ Control(S.dagger) @ I)
  u << (I @ H @ get_I(2))
  u << swap12

  QPE = u

  ''' RY '''
  swap01 = SWAP @ get_I(2)
  u =  swap01 * (Control(RY(r1)) @ get_I(2)) * swap01
  u << swap12
  u << swap01 * (Control(RY(r2)) @ get_I(2)) * swap01
  u << swap12

  CR = u

  ''' iQPE '''
  iQPE = QPE.dagger

  ''' final state '''
  return QPE << CR << iQPE | enc_b


#params = [pi, pi/2, pi/8, pi/16]
params = [2*pi, pi, pi/8, pi/16]
delta = 0.01

last_prob = 0.0
for s in range(4000):
  p_id = randrange(len(params))

  v_old = params[p_id]

  if random() < 0.5:
    params[p_id] = v_old + delta
  else:
    params[p_id] = v_old + delta

  q: State = HHL(A12, b1, *params)
  
  print(np.abs(q.v).sum())

  good_prob = np.abs(q.v)[8:].sum()

  if good_prob > last_prob:
    last_prob = good_prob
  else:
    params[p_id] = v_old

  if s % 100 == 0:
    print(params)

print('last_prob:', last_prob)
print('params:', params)


questions = [
  (A12, b1),
  (A12, b0),
  (A12, bp),
  (A12, bn),
  (A13, b1),
  (A13, b0),
  (A13, bp),
  (A13, bn),
  (A23, b1),
  (A23, b0),
  (A23, bp),
  (A23, bn),
]

for A, b in questions:
  x = HHL(A, b, *params).v.real
  z = np.linalg.solve(A, b)
  print('x:', [x[0], x[8]])
  print('z:', z)
  print()
