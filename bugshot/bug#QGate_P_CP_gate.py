#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/04 

from pyqpanda import *
import numpy as np

# BUG: CPGate not work, and failed to print PGate parameter value in circuit printer
# location: QPanda-2\include\Core\Utilities\QProgInfo\QCircuitInfo.h:495-503
# expected: Can print the parameter value of the PGate
# actually: get warning `QPanda::get_gate_parameter Unsupported GateNode`

# FIXME: 
#   1. CPGate does not show in circuit printer, I do not know why
#   2. the swicth statement of `Core\Utilities\Compiler\QProgToOriginIR.cpp:207` forgets to handle with `P_GATE`

assert P
assert CP

# ↓↓↓ Example ↓↓↓

qvm = CPUQVM()
qvm.init_qvm()

q = qvm.qAlloc_many(2)

cq = QCircuit() \
   << CP(q[0], q[1], np.pi) \
   << P(q[0], np.pi) \
   << CP(q[1], q[0], np.pi) \
   << P(q[1], np.pi)

print(cq)
