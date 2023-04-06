#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/04 

from pyqpanda import *
import numpy as np

# BUG: assign() does not work inplace, cannot be used together with QWhile for a times-known iteration 
# expected: use `assign()` in QWhile to count the iteration steps
# actually: stcuk into a deadloop

# FIXME: check the semantics of `assign()` implementation

assert assign

# ↓↓↓ Example ↓↓↓

qvm = CPUQVM()
qvm.init_qvm()

q = qvm.qAlloc_many(1)
c = qvm.cAlloc_many(2)
c[0].set_val(0)   # value
c[1].set_val(0)   # counter

# NOTE: this prog should apply RY for 10 times, but stuck dead-loop forever
prog = QProg() \
  << RY(q[0], np.pi/7) \
  << assign(c[1], c[1] + 1)     # FIXME: should do a inplcae c[1] <- c[1] + 1
qwhile = create_while_prog(c[1] < 10, prog)

prog = QProg() << qwhile << Measure(q[0], c[0])
ir = to_originir(prog, qvm)
print(ir)                       # FIXME: compiled code is also failed to apply `c[1]+1`
print()

print('start [directly_run]')
result = qvm.directly_run(prog)
print('finish [directly_run]')    # <= non-reaching
print(result)
print('value:',   c[0].get_val())
print('counter:', c[1].get_val())
