#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/02 

from pyqpanda import *

# BUG: this API does not return as expected
# location: QPanda-2\include\Core\QuantumCircuit\QProgram.h:78-82
# expected: Measure operation in the last position of the program
# actually: alway return True

assert QProg.is_measure_last_pos

# ↓↓↓ Example ↓↓↓

qvm = CPUQVM()
qvm.init_qvm()
q = qvm.qAlloc_many(2)
c = qvm.cAlloc_many(2)

progs = [
  # empty prog
  QProg(),                                            # <= expect False
  QProg() << Measure(q[0], c[0]),
  QProg() << measure_all(q, c),
  QProg() << meas_all(q, c),
  QProg() << H(q[0]),                                 # <= expect False
  QProg() << Measure(q[0], c[0]) << H(q[0]),          # <= expect False
  QProg() << measure_all(q, c)   << H(q[0]),          # <= expect False
  QProg() << meas_all(q, c)      << H(q[0]),          # <= expect False

  # prog with 1 qubit
  QProg() << H(q[0]) << Measure(q[0], c[0]),
  QProg() << H(q[0]) << Measure(q[0], c[0]) << H(q),  # <= expect False

  # prog with 2 qubit2
  QProg() << H(q),                                    # <= expect False
  QProg() << H(q) << measure_all(q, c),
  QProg() << H(q) << meas_all(q, c),
  QProg() << H(q) << measure_all(q, c) << H(q),       # <= expect False
  QProg() << H(q) << meas_all(q, c)    << H(q),       # <= expect False
]

for prog in progs:
  prog: QProg
  print(prog.is_measure_last_pos())
