#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/02 

from pyqpanda import *

qvm = CPUQVM()
qvm.init_qvm()


def test_single_qubit():
  q0 = qvm.qAlloc()
  c0 = qvm.cAlloc()

  qc = QCircuit() \
     << Y(q0) \
     << H(q0)
  print(qc)

  prog = QProg() << qc << Measure(q0, c0)
  print(qvm.get_prob_dict([q0]))
  for _ in range(10):
    print(qvm.directly_run(prog))


def test_single_qubit1():
  q0 = qvm.qAlloc()
  c0 = qvm.cAlloc()

  prog = QProg() \
     << H(q0) \
     << Measure(q0, c0)
  print(prog)

  print(qvm.get_prob_list([q0]))
  print(qvm.get_prob_tuple_list([q0]))
  print(qvm.get_prob_dict([q0]))
  print(qvm.pmeasure([q0]))
  print(qvm.pmeasure_no_index([q0]))
  print(qvm.quick_measure([q0], 1000))

  print(qvm.prob_run_dict(prog, [q0]))
  print(qvm.prob_run_tuple_list(prog, [q0]))
  print(qvm.prob_run_list(prog, [q0]))
  print(qvm.run_with_configuration(prog, [c0], 1000))

  for _ in range(10):
    print(qvm.async_run(prog))
    print(qvm.directly_run(prog))


def test_multi_qubits():
  q1, q0 = qvm.qAlloc_many(2)
  c1, c0 = qvm.cAlloc_many(2)

  qc = QCircuit() \
     << Y(q1) \
     << CNOT(q1, q0) \
     << H(q1)
  print(qc)

  prog = QProg() << qc << measure_all([q1, q0], [c1, c0])
  #print(qvm.get_prob_dict([q1, q0]))
  for _ in range(10):
    print(qvm.directly_run(prog))


if __name__ == '__main__':
  test_single_qubit1()
  test_single_qubit()
  test_multi_qubits()
