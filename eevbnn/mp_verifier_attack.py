"""attack by the verifier in multiple processes

This script starts a server and receive requests on stdin
"""

from .utils import setup_pyx_import, torch_as_npy, ensure_training_state
from .eval_bin import ModelVerifier, init_argparser as init_verifier_argparser
with setup_pyx_import():
    from ._atomic import AtomicInt32

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

import os
import sys
import collections
import subprocess
import tempfile
import time
import argparse
from multiprocessing.shared_memory import SharedMemory

RemoteArgs = collections.namedtuple(
    'RemoteArgs',
    ['eval_model', 'inputs', 'inputs_adv_ref', 'labels',
     'correct_mask', 'epsilon', 'max_nr_adv', 'shm_name', 'shm_size',
     'idx_remap']
)

BUFFER_COUNTER_ELEM_SIZE = 128  # pad to cache line
BUFFER_COUNTER_NR = 4   # read_idx, done_workers, nr_adv_verifier, nr_adv_ref
BUFFER_COUNTER_SIZE = BUFFER_COUNTER_ELEM_SIZE * BUFFER_COUNTER_NR

DEFAULT_TIMEOUT = 2

class WorkerImpl:
    _args = None
    _progress_cb = None
    _verifier = None

    _read_idx: AtomicInt32 = None   # negative value for error
    _done_workers: AtomicInt32 = None
    _nr_adv_verifier: AtomicInt32 = None
    _nr_adv_ref: AtomicInt32 = None

    _output: np.ndarray = None

    def __init__(self, args: RemoteArgs, progress_cb, is_master):
        self._args = args
        self._progress_cb = progress_cb
        self._make_verifier()

        shm = SharedMemory(args.shm_name, False)
        try:
            self._setup_shm(shm.buf)
            self._process()
            self._done_workers.fetch_add(1)
        except:
            # set to a negative value to mark failure
            self._read_idx.fetch_add(-args.inputs.shape[0]**2)
            raise
        finally:
            self._read_idx = None
            self._done_workers = None
            self._nr_adv_verifier = None
            self._nr_adv_ref = None
            self._output = None
            shm.close()
            # manual cleanup due to a bug (https://bugs.python.org/issue39959 )
            if not is_master and sys.version_info <= (3, 8, 2):
                from multiprocessing import shared_memory
                if shared_memory._USE_POSIX:
                    from multiprocessing.resource_tracker import unregister
                    unregister(shm._name, "shared_memory")

    def _process(self):
        args = self._args
        inputs_npy = torch_as_npy(args.inputs)
        inputs_adv_ref_npy = torch_as_npy(args.inputs_adv_ref)
        labels_npy = torch_as_npy(args.labels)
        batch_size = inputs_npy.shape[0]
        while True:
            nr_done = idx = self._read_idx.fetch_add(1)
            if idx >= batch_size or idx < 0:
                self._read_idx.fetch_add(-1)
                return
            idx = args.idx_remap[idx]
            nr_adv_verifier = self._nr_adv_verifier.load()
            if nr_adv_verifier >= args.max_nr_adv or not args.correct_mask[idx]:
                adv = None
                if not args.correct_mask[idx]:
                    self._nr_adv_ref.fetch_add(1)
            else:
                _, _, adv = self._verifier.check(
                    inputs_npy[idx], labels_npy[idx])
                if adv is not None:
                    self._nr_adv_verifier.fetch_add(1)

            if adv is None:
                adv = inputs_adv_ref_npy[idx]
            self._output[idx] = adv
            if self._progress_cb is not None:
                self._progress_cb(
                    nr_done,
                    self._nr_adv_verifier.load(), self._nr_adv_ref.load())

    def _setup_shm(self, buf):
        counter_buf = (np.
                       frombuffer(buf, np.int32, BUFFER_COUNTER_SIZE // 4).
                       reshape(BUFFER_COUNTER_NR, -1))
        (self._read_idx, self._done_workers,
         self._nr_adv_verifier, self._nr_adv_ref) = map(
             AtomicInt32, counter_buf)
        result_buf = buf[BUFFER_COUNTER_SIZE:self._args.shm_size]
        self._output = (np.frombuffer(result_buf, np.float32).
                        reshape(self._args.inputs.shape))

    def _make_verifier(self):
        verifier_args = (init_verifier_argparser(argparse.ArgumentParser()).
                         parse_args(args=[]))
        verifier_args.eps = self._args.epsilon
        verifier_args.timeout = DEFAULT_TIMEOUT
        self._verifier = ModelVerifier(verifier_args, [self._args.eval_model])
        self._verifier.enable_score_check = False
        self._verifier.log = lambda _: None


class VerifierAdvBatchFinder:
    """find adversarial examples for a training batch by a verifier"""

    _workers = None
    _nr_workers = None

    def __init__(self, nr_workers):
        assert nr_workers >= 1
        self._nr_workers = nr_workers

    def __del__(self):
        self.close()

    def _start_workers(self):
        if self._workers is None:
            env = dict(os.environ)
            env['CUDA_VISIBLE_DEVICES'] = ''    # disable cuda in the workers
            self._workers = [
                subprocess.Popen(
                    [sys.executable, '-m', 'eevbnn', 'mp_verifier_attack'],
                    stdin=subprocess.PIPE,
                    env=env,
                )
                for _ in range(self._nr_workers - 1)
            ]

    def find_adv_batch(
            self,
            model: nn.Module,
            inputs: torch.Tensor, inputs_adv_ref: torch.Tensor,
            labels: torch.Tensor, epsilon: float, max_nr_adv):

        self._start_workers()

        with ensure_training_state(model, False):
            model_outputs: torch.Tensor = model(inputs_adv_ref)
            eval_model = model.cvt_to_eval()

        correct_mask = torch.eq(model_outputs.argmax(dim=1), labels)
        idx_remap = np.arange(inputs.shape[0], dtype=np.int32)
        np.random.shuffle(idx_remap)
        assert inputs.dtype == torch.float32

        shm_size = BUFFER_COUNTER_SIZE + 4 * inputs.numel()
        shm = SharedMemory(size=shm_size, create=True)
        shm.buf[:BUFFER_COUNTER_SIZE] = b'\0' * BUFFER_COUNTER_SIZE
        shm_name = shm.name

        args = RemoteArgs(eval_model, inputs, inputs_adv_ref,
                          labels, correct_mask, epsilon, max_nr_adv,
                          shm_name, shm_size, idx_remap)

        try:
            return self._work(shm.buf, args)
        except:
            self.close()
            raise
        finally:
            shm.close()
            shm.unlink()

    def _extract_return_value(self, buf, args):
        counter_buf = (np.
                       frombuffer(buf, np.int32, BUFFER_COUNTER_SIZE // 4).
                       reshape(BUFFER_COUNTER_NR, -1))

        # wait for finish
        read_idx, done_workers = map(AtomicInt32, counter_buf[:2])
        wait_begin = time.time()
        while done_workers.load() < self._nr_workers:
            assert read_idx.load() >= args.inputs.shape[0]
            time.sleep(0.05)
            self._check_alive()
            if time.time() - wait_begin > 1:
                print(
                    f'waiting for workers: '
                    f'done={done_workers.load()}/{self._nr_workers} '
                    f'read_idx={read_idx.load()}/{args.inputs.shape[0]}  ')
                wait_begin = time.time()

        read_idx, _, nr_adv_verifier, nr_adv_ref = (
            int(i[0]) for i in counter_buf)
        result_buf = buf[BUFFER_COUNTER_SIZE:args.shm_size]
        result = (np.
                  frombuffer(result_buf, np.float32).
                  reshape(args.inputs.shape))
        assert read_idx >= args.inputs.shape[0], 'read idx indicates failure'

        result = torch.from_numpy(result.copy()).to(args.inputs.device)
        return result, nr_adv_verifier, nr_adv_ref

    def _check_alive(self):
        for i in self._workers:
            assert i.poll() is None, 'worker died'

    def _work(self, shm_buf, args):
        prev_time = 0

        def report_progress(idx, nr_adv_verifier, nr_adv_ref):
            nonlocal prev_time

            now = time.time()
            pbar.n = idx
            pbar.set_description(f'adv={nr_adv_verifier}', False)
            if now - prev_time >= 0.1:
                pbar.refresh()
                prev_time = now

        def work_self():
            WorkerImpl(args, report_progress, True)
            return self._extract_return_value(shm_buf, args)

        with tqdm(total=args.inputs.shape[0]) as pbar:
            if self._workers:
                with tempfile.NamedTemporaryFile('wb') as fout:
                    torch.save(args, fout)
                    fout.flush()
                    for i in self._workers:
                        i.stdin.write(fout.name.encode('utf-8'))
                        i.stdin.write(b'\n')
                        i.stdin.flush()
                    return work_self()
            else:
                return work_self()


    def close(self):
        """close the worker processes"""
        if self._workers is not None:
            for i in self._workers:
                i.kill()
            for i in self._workers:
                i.wait()
            del self._workers

def main():
    while True:
        fname = sys.stdin.readline().strip()
        with open(fname, 'rb') as fin:
            args = torch.load(fin, 'cpu')
        WorkerImpl(args, None, False)

if __name__ == '__main__':
    main()
