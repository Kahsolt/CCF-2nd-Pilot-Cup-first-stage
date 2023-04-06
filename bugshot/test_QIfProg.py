#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/03 

from pyqpanda import *

qvm = CPUQVM()
qvm.init_qvm()
qubits = qvm.qAlloc_many(3)
cbits = qvm.cAlloc_many(3)
cbits[0].set_val(0)
cbits[1].set_val(3)

# 构建QIf
qif = QIfProg(cbits[0] > cbits[1] and cbits[0] < cbits[1],
              QProg() << H(qubits[0])<< H(qubits[1]) << H(qubits[2]), 
              QProg() << H(qubits[0]) << CNOT(qubits[0], qubits[1]) << CNOT(qubits[1], qubits[2]))

# QIf插入到量子程序中
prog = QProg() << qif

print(prog)

# 概率测量，并返回目标量子比特的概率测量结果，下标为十进制
result = qvm.prob_run_dict(prog, qubits, -1)

# 打印概率测量结果
print(result)
