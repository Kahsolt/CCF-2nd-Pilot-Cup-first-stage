#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/01 

D = [0, 1, 2]

f_A = lambda x, y: (y, x)
f_B = lambda x, y: ((x + y) % 3, (x * y) % 3)
f_C = lambda x, y: ((x + y) % 3, (x - y) % 3)
f_D = lambda x, y: ((x + y) % 3, (2*x - y) % 3)

print('The non-invertibles are:')
for name in ['A', 'B', 'C', 'D']:
  f = globals()[f'f_{name}']
  
  res = {}
  is_dup = False
  for x in D:
    if is_dup: break
    for y in D:
      if is_dup: break

      o = f(x, y)
      if o in res:
        is_dup = True
        break
      else:
        res[o] = (x, y)
  
  if is_dup: print(name)
