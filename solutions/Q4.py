#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/01 

from tiny_q import *

u = (X @ 3) * CCNOT * (X @ X @ I)

for s in ['00', '01', '10', '11']:
  q = v(s) @ v0
  res = u | q > Measure(300)
  cnt_val = [(cnt, val) for val, cnt in res.items()]
  cnt_val.sort(reverse=True)
  top_val = cnt_val[0][1]
  print(f'f({s}) = {top_val[2:]}')
