#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/01 

from tiny_q import *

def make_GHZ(n:int=3):
  assert n > 0
  
  u = H @ get_I(n-1)
  for j in range(n-1):
    u = (get_I(j) @ CNOT @ get_I(n-j-2)) * u

  return u


for j in range(1, 10):
  u = make_GHZ(j)
  q = v('0' * j)
  res = u | q > Measure()
  print({k: v for k, v in res.items() if v > 0})
