#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/04 

from pyqpanda import *
import numpy as np

# (2) API
assert QWhileProg
QWhileProg.get_classical_condition
QWhileProg.get_true_branch


''' Test '''
qvm = CPUQVM()
qvm.init_qvm()

q = qvm.qAlloc_many(1)
c = qvm.cAlloc_many(2)
c[0].set_val(0)   # coin tossing result
c[1].set_val(0)   # counter

# NOTE: a very unfair coin: P(0) = 95%
coin = QProg() \
  << RY(q[0], np.pi/7) \
  << assign(c[1], c[1] + 1) \
  << measure_all(q, [c[0]])         # measure q[0] => c[0]
# toss the coin until it gets an `1``
prog = QProg() << QWhileProg(c[0] == 0, coin)

print('prog:')
print(prog)
qvm.directly_run(prog)
print(qvm.get_qstate())
print('coin:',    c[0].get_val())   # expect value == 1
print('counter:', c[1].get_val())   # FIXME: expect value > 0
