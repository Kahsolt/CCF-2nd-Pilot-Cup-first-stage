#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/04 

from pyqpanda import *
import numpy as np

# BUG: QWhileProg is broken, even the official example does NOT yield the shown results
# location: https://pyqpanda-tutorial-en.readthedocs.io/en/latest/chapter2/index.html#id4
# expected: work properly
# actually: copied the demo code, ran, did not yield the demo results; QWhileProg does not work

# NOTE: here's another simpel example to show QWhileProg DOES NOT EVEN WORK!!

assert QWhileProg

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
