#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/08

import numpy as np
from pyqpanda import *

qvm = CPUQVM()
qvm.init_qvm()

import P2 ; P2.qvm = qvm
from P2 import HHL_2x2_qpanda, A12, b1

qc = HHL_2x2_qpanda(A12, b1, ver='QPanda-dump-5')
print(qc)

qvm.directly_run(qc)
print(np.round(qvm.get_qstate(), 5).real)
qvm.finalize()
