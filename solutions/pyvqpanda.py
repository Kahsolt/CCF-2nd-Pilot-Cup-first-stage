#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/02 

# Roarrrr!! PyQPanda's API is a fucking mess, just wrap it again!!

from pyqpanda import *
from typing import List, Tuple, Dict, Union

Cbit = ClassicalCondition


class VQVM():

  ''' The virtual again universal interface for most of the `pyqpanda.QuantumMachine` :) '''

  def __init__(self, qvm_cls=CPUQVM):
    assert issubclass(qvm_cls, QuantumMachine), 'Oh you fool got a fake QuantumMachine!'
    assert qvm_cls != DensityMatrixSimulator, 'DensityMatrixSimulator is not yet supported!'
    assert qvm_cls != QCloud, 'We do not own a QCloud!'

    self.qvm = qvm_cls()

  def __enter__(self):
    self.startup()

  def __exit__(self, exc_type, exc_value, exc_traceback):
    self.shutdown()

  def startup(self):
    self.qvm.init_qvm()

  def shutdown(self):
    self.qvm.finalize()

  def status(self):
    print('================')
    print('type:', self.qvm.__class__.__name__)
    print('qubits:', self.qvm.get_allocate_cmem_num())
    print('cbits:', self.qvm.get_allocate_qubit_num())
    print('qstate', self.qvm.get_qstate())
    print('================')
    #print('Status', self.qvm.get_status())   # This API is broken

  def alloc_qubit(self, cnt=1) -> Union[Qubit, List[Qubit]]:
    assert isinstance(cnt, int) and cnt >= 1
    return self.qvm.qAlloc() if cnt == 1 else self.qvm.qAlloc_many(cnt)

  def alloc_cbit(self, cnt=1) -> Union[Cbit, List[Cbit]]:
    assert isinstance(cnt, int) and cnt >= 1
    return self.qvm.cAlloc() if cnt == 1 else self.qvm.cAlloc_many(cnt)

  def free_qubit(self, q:Union[Qubit, List[Qubit]]):
    if isinstance(q, Qubit): self.qvm.qFree(q)
    if isinstance(q, list):  self.qvm.qFree_all(q)

  def free_cbit(self, c:Union[Cbit, List[Cbit]]):
    if isinstance(c, Cbit): self.qvm.cFree(c)
    if isinstance(c, list): self.qvm.cFree_all(c)

  def pmeasure(self, prog:QProg, q:Union[QVec, List[Qubit]], type:Union[dict, tuple, list]=dict, select_max:int=-1, idx:Union[int, str, List[int], List[str]]=None) \
      -> Union[Dict[str, float], List[Tuple[int, float]], List[float], complex, List[complex]]:
    '''
      PMeasure runs a QProg and returns the theoretical probability distribution or the amplitude
      NOTE: the given QProg should NOT contain any Measure node
    '''

    # TODO: has_measure_gate(prog:QProg) -> bool
    it = prog.begin()
    while it != prog.end():
      assert it.get_node_type() != NodeType.MEASURE_GATE
      it = it.get_next()

    if idx is not None:
      assert len(idx), 'index cannot be an empty list'
      if isinstance(idx, list):
        assert isinstance(self.qvm, (PartialAmpQVM, MPSQVM)), 'only PartialAmpQVM and MPSQVM support measuring partial qubits at given indexes'
        if isinstance(idx[0], int):
          return self.qvm.pmeasure_dec_subset(prog, [str(i) for i in idx])
        elif isinstance(idx[0], str):
          return self.qvm.pmeasure_bin_subset(prog, idx)
        else:
          raise TypeError(f'invalid type for idx, got {type(idx[0])}')
      elif isinstance(idx, (int, str)):
        assert isinstance(self.qvm, (SingleAmpQVM, PartialAmpQVM, MPSQVM)), 'only SingleAmpQVM, PartialAmpQVM and MPSQVM support measuring single qubit at given index'
        if isinstance(idx, int):
          return self.qvm.pmeasure_dec_index(prog, [str(i) for i in idx])
        elif isinstance(idx, str):
          return self.qvm.pmeasure_bin_index(prog, idx)
      else:
        raise TypeError(f'invalid type for idx, got {type(idx)}')
    else:
      runner = {
        dict:  lambda: self.qvm.prob_run_dict,
        tuple: lambda: self.qvm.prob_run_tuple_list,     # i.e. qvm.pmeasure
        list:  lambda: self.qvm.prob_run_list,           # i.e. qvm.pmeasure_no_index
      }
      return runner[type]()(prog, q, select_max)

  def measure(self, prog:QProg, cnt=1000, noise:Noise=None) -> Dict[str, int]:
    '''
      Measure runs a QProg and returns the result or results counter
        - one-shot measure
          - QProg with Measure + directly_run()
        - multi-shot measure
          - QProg with Measure + run_with_configuration()
          - QProg without Measure + directly_run() + quick_measure(); the deprecated old interface
    '''

    if 'prog does not contain Measure':
      self.qvm.directly_run(prog, noise_model=noise)
      return self.qvm.quick_measure()
    else:
      # run_with_configuration := measure_all + directly_run + quick_measure
      # NOTE: optional parameter `cbit_list` is not necessary, and does not even work!!
      return self.qvm.run_with_configuration(prog, shot=cnt, noise_model=noise)


if __name__ == '__main__':
  vqvm = VQVM()
  vqvm.startup()

  vqvm.status()

  qs = vqvm.alloc_qubit(3)
  cs = vqvm.alloc_cbit(2)
  vqvm.status()

  vqvm.free_qubit(qs)
  vqvm.free_cbit(cs)

  vqvm.shutdown()
