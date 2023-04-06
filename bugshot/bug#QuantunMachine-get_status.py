#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/02 

from pyqpanda import *
from traceback import print_exc

# BUG: these APIs are broken
# location: QPanda-2\include\Core\QuantumMachine\QuantumMachineInterface.h:146-150
# expected: Get the status of the Quantum machine
# actually: raise TypeError
assert QuantumMachine.get_status
assert QuantumMachine.getStatus

# ↓↓↓ Example ↓↓↓

qvm = CPUQVM()
qvm.init_qvm()

try:
  # QuantumMachine.get_status(qvm)
  qvm.get_status()    # <= this API is broken
except TypeError:
  print_exc()

try:
  # QuantumMachine.getStatus(qvm)
  qvm.getStatus()     # <= this API is broken
except TypeError:
  print_exc()
