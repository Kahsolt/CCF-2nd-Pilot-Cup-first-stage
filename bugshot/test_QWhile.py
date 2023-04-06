#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/04 

from pyqpanda import *


init(QMachineType.CPU)
qubits = qAlloc_many(3)
cbits = cAlloc_many(3)
cbits[0].set_val(0)
cbits[1].set_val(1)

prog_while = QProg() \
  << H(qubits[0]) \
  << H(qubits[1]) \
  << H(qubits[2]) \
  << assign(cbits[0], cbits[0] + 1) \
  << Measure(qubits[1], cbits[1])

qwhile = create_while_prog(cbits[1], prog_while)

prog = QProg() << qwhile
result = directly_run(prog)
print(result)
print(cbits[0].get_val())
print(cbits[1].get_val())
finalize()
