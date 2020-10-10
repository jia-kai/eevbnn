#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# so X11 would not be needed
import matplotlib
matplotlib.use('Agg')

import torch
from eevbnn.net_bin import ModelHelper, BiasRegularizer
from eevbnn.utils import default_dataset_root, ensure_dir
from eevbnn.net_tools import eval_cbd_stats

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import copy
import argparse
import os
import re
import functools
import collections
import json
import itertools
import zlib
import weakref
from pathlib import Path

ADV2_255 = '0.00784313725490196'
ADV20_255 = '0.0784313725490196'
ADV5_255 = '0.0196078431372549'
ADV8_255 = '0.03137254901960784'

# chosen cbd config for mnist and cifar10
CHOSEN_CBD = ['3-vrf', '3-vrf']

COLOR4 = [
    '#a6cee3',
    '#1f78b4',
    '#b2df8a',
    '#33a02c',
]

g_possible_tested_cases = set()

# patch latex formatter to output cline for single rows
import pandas.io.formats.latex
class FixedClineLatexFormatter(pandas.io.formats.latex.LatexFormatter):
    def _format_multirow(self, row, ilevels, i, rows):
        for j in range(ilevels):
            if row[j].strip():
                nrow = 1
                for r in rows[i + 1 :]:
                    if not r[j].strip():
                        nrow += 1
                    else:
                        break
                if nrow > 1:
                    # overwrite non-multirow entry
                    row[j] = "\\multirow{{{nrow:d}}}{{*}}{{{row:s}}}".format(
                        nrow=nrow, row=row[j].strip()
                    )
                if j < ilevels - 1:
                    # save when to end the current block with \cline
                    self.clinebuf.append([i + nrow - 1, j + 1])
        return row
pandas.io.formats.latex.LatexFormatter = FixedClineLatexFormatter

def gen_prod_treelike(desc):
    """recursively generate product for tree-like description"""
    if isinstance(desc[0], (list, tuple)):
        ret = []
        for k, v in desc:
            assert isinstance(k, str)
            for i in gen_prod_treelike(v):
                ret.append([k] + i)
        return ret
    return [[i] for i in desc]

def fix_cline(latex):
    """use cmidrule for cline"""
    return latex.replace('cline', 'cmidrule(lr)')

def add_second_row_title(latex: str, title: str) -> str:
    """add a title to the second row"""
    lines = latex.split('\n')
    cols = lines[3].split('&')
    cols[0] = title
    lines[3] = '&'.join(cols)

    cols = lines[2].split('&')
    cidx = 0
    for i, j in list(enumerate(cols)):
        j = j.strip()
        cidx += 1
        if (m := re.search(r'multicolumn\{([0-9]*)\}', j)) is not None:
            cnext = cidx + int(m.group(1)) - 1
            cols[-1] += r' \cmidrule(lr){%d-%d}' % (cidx, cnext)
            cidx = cnext
    lines[2] = '&'.join(cols)
    return '\n'.join(lines)

class PercentValue:
    __slots__ = ['vnum', 'precision', 'empty_for_zero']

    def __init__(self, s, precision=0):
        self.precision = precision
        if isinstance(s, str):
            s = s.strip()
            assert s.endswith('%')
            self.vnum = float(s[:-1]) / 100
        else:
            self.vnum = float(s)
        self.empty_for_zero = False

    def set_precision(self, prec):
        self.precision = prec
        return self

    def set_empty_for_zero(self, flag=True):
        self.empty_for_zero = flag
        return self

    def neg(self):
        """1 - self"""
        return PercentValue(1 - self.vnum, self.precision)

    def __repr__(self):
        if self.empty_for_zero and not self.vnum:
            return '0'
        val = self.vnum * 100
        fmt = '{{:.{}f}}\\%'.format(self.precision).format
        ret = fmt(val)
        if val != 0 and ret == fmt(0.0):
            # handle small numbers
            return r'{:.1g}\%'.format(val)
        return ret


class FloatValue:
    def __init__(self, s, precision):
        self.precision = precision
        self.vnum = float(s)

    def copy(self):
        return FloatValue(self.vnum, self.precision)

    def set_precision(self, prec):
        self.precision = prec
        return self

    def _format_latex_sci(self):
        if self.vnum == 0:
            return '0'
        sci = '{:.1e}'.format(self.vnum)
        k, e = sci.split('e')

        assert e[0] in '+-'
        if e[1] == '0':
            e = e[0] + e[2:]
        if e[0] == '+':
            e = e[1:]

        if k.endswith('.0'):
            k = k[:-2]
        return fr'\scinum{{ {k} }}{{ {e} }}'

    def __repr__(self):
        p = self.precision
        if p == 'latex-sci':
            return self._format_latex_sci()

        if isinstance(p, str):
            fmt = '{:' + p + '}'
        else:
            assert isinstance(p, int), f'bad precision {p}'
            fmt = '{{:.{}f}}'.format(p)

        ret = fmt.format(self.vnum)
        if ret == fmt.format(0.0) and abs(self.vnum) > 1e-9:
            return f'${self._format_latex_sci()}$'
        return ret

    def __add__(self, rhs):
        assert isinstance(rhs, FloatValue) and rhs.precision == self.precision
        return FloatValue(self.vnum + rhs.vnum, self.precision)

    def __truediv__(self, rhs):
        assert isinstance(rhs, FloatValue), rhs
        return self.vnum / rhs.vnum


def cached_meth(fn):
    """decorator for cached class method without arguments"""
    key = f'__result_cache_{fn.__name__}'

    @functools.wraps(fn)
    def work(self):
        d: dict = self.__dict__
        ret = d.get(key)
        if ret is not None:
            return ret
        ret = fn(self)
        assert ret is not None
        d[key] = ret
        return ret
    return work

SparsityStat = collections.namedtuple(
    'SparsityStat',
    ['layerwise', 'tot_avg', 'nr_non_zero']
)
SolverStat = collections.namedtuple(
    'SolverStat',
    ['num', 'build_time',
     'solve_time', 'solve_time_mid', 'solve_time_min', 'solve_time_max',
     'solve_prob', 'robust_prob']
)

class SingleExperiment:
    _args = None
    _data_dir = None
    _all_experiments = None
    _name = None

    _used_solver_stats = None
    """map from key to bool indicating whether the full set is used"""

    def __init__(self, args, data_dir: Path, all_experiments):
        assert isinstance(all_experiments, Experiments)
        self._args = args
        self._data_dir = data_dir
        self._all_experiments = weakref.proxy(all_experiments)
        self._name = data_dir.name
        self._used_solver_stats = collections.defaultdict(bool)

    def _open_log(self):
        return (self._data_dir / 'log.txt').open()

    @property
    def name(self):
        return self._name

    @cached_meth
    def _check_train_finish(self):
        assert (self._data_dir / 'finish_mark').exists(), (
            f'training of {self._data_dir} is unfinished')
        return True

    @property
    @cached_meth
    def _log_lines(self):
        with self._open_log() as fin:
            return fin.readlines()

    def _check_format_v2(self, prefix, default_idx):
        """find value with specifid prefix in v2 format log"""
        if (self._log_lines[-1].startswith('model choosing') or
                self._log_lines[-2].startswith('model choosing')):
            # log format v2 with model choosing
            UPDATE_MAGIC = 'best model update at '
            lookfor = None
            logline = None
            for i in self._log_lines[::-1]:
                if lookfor is not None and i.startswith(lookfor):
                    return i
                if i.startswith(UPDATE_MAGIC):
                    lookfor = prefix + i[len(UPDATE_MAGIC):].strip()
            raise RuntimeError(f'{lookfor} not found in log')
        return self._log_lines[default_idx]

    @cached_meth
    def sparsity(self):
        self._check_train_finish()
        logline = self._check_format_v2('sparsity', -2)
        m = re.match(r'sparsity@[0-9]*: (.*) ; ([0-9]*)/([0-9]*)=.*', logline)
        layer = list(map(PercentValue, m.group(1).split(',')))
        nz, tot = map(int, [m.group(2), m.group(3)])
        return SparsityStat(layer, PercentValue(nz / tot), tot - nz)

    @cached_meth
    def test_acc(self):
        self._check_train_finish()
        logline = self._check_format_v2('test', -1)
        m = re.match(r'test@[0-9]*: acc=([0-9.]*%)', logline)
        assert m is not None, (
            f'bad logline: {logline.strip()} @ {self._data_dir}')
        return PercentValue(m.group(1), precision=2)

    @cached_meth
    def test_loss(self):
        self._check_train_finish()
        logline = self._check_format_v2('test', -1)
        m = re.match(r'test@[0-9]*: .* loss=([0-9.]*) ', logline)
        assert m is not None, (
            f'bad logline: {logline.strip()} @ {self._data_dir}')
        return FloatValue(m.group(1), precision=2)

    def ensemble_test_acc(self, eps):
        fpath = self._data_dir / f'eval-ensemble-{eps}.json.ensemble_acc'
        with fpath.open() as fin:
            return PercentValue(float(fin.read()), 2)

    @cached_meth
    def _all_solver_raw(self):
        self._check_train_finish()
        check_merged_all = collections.defaultdict(dict)
        all_result = {}
        for jfile in self._data_dir.glob('eval-*.json'):
            param_class = jfile.name.split('-')[-1][:-5]    # eps
            float(param_class)  # ensure it is valid
            if '-ensemble-' in jfile.name:
                param_class = f'ensemble-{param_class}'
            check_merged = check_merged_all[param_class]

            jf_path = self._data_dir / f'{jfile.name}.finished'

            with jfile.open() as fin:
                data = json.load(fin)
            all_result[jfile.name] = data
            for k, v in data.items():
                old = check_merged.setdefault(k, v['result'])
                if old != v['result']:
                    assert old == 'TLE' or v['result'] == 'TLE', (
                        f'verify result mismatch on {jfile}: {old=} {k=} {v=}'
                    )
                    if old == 'TLE':
                        check_merged[k] = v['result']

            if not jf_path.exists():
                if self._args.add_unfinished:
                    print(f'WARNING: {jf_path} does not exist but still added')
                else:
                    del all_result[jfile.name]

        return all_result

    def solver_stats(self, eps, name=None, *, use_subset=False,
                     default_name='minisatcs-verify'):
        """
        :param use_subset: if True, use only the sub test set of this dataset
        """
        name = name or default_name

        try:
            key = f'eval-{name}-{eps}.json'
            data: dict = self._all_solver_raw()[key]
        except KeyError:
            raise RuntimeError(f'{key} not found in expr {self._data_dir}')

        if use_subset:
            data = {k: v for k, v in data.items()
                    if k in self._solver_stats_subset_ref()}

            self._used_solver_stats[key]    # insert the key
        else:
            self._used_solver_stats[key] = True

        all_build_times = []
        all_solve_times = []
        is_solved = []
        is_robust = []

        result2int = {
            'UNSAT': 0,
            'SAT': 1,
            'TLE': -1
        }

        for i in data.values():
            all_build_times.append(i['build_time'])
            all_solve_times.append(i['solve_time'])
            result = result2int[i['result']]
            is_solved.append(result >= 0)
            is_robust.append(result == 0)

        checksum = hex(
            zlib.adler32(','.join(sorted(data.keys())).encode('utf-8')))
        g_possible_tested_cases.add(f'{len(data)}[{checksum}]')

        prec = 0
        prec_num = 100
        while len(data) > prec_num:
            prec += 1
            prec_num *= 10

        return SolverStat(
            len(data),
            FloatValue(np.mean(all_build_times), 3),
            FloatValue(np.mean(all_solve_times), 3),
            FloatValue(np.median(all_solve_times), 3),
            FloatValue(np.min(all_solve_times), 3),
            FloatValue(np.max(all_solve_times), 3),
            PercentValue(np.mean(is_solved), prec),
            PercentValue(np.mean(is_robust), prec),
        )

    @cached_meth
    def cbd_stats(self):
        """:return: cbd loss, cbd avg, cbd max"""
        cache_file = self._data_dir / 'cbd_stat.json'
        if cache_file.exists():
            with cache_file.open() as fin:
                cbd_avg, cbd_max = json.load(fin)
        else:
            cbd_avg, cbd_max = self._eval_cbd_stats()
            with cache_file.open('w') as fout:
                json.dump([cbd_avg, cbd_max], fout)

        m = re.search('bias_hinge_coeff=([^,]*),', self._log_lines[1])
        coeff = float(m.group(1))

        return (FloatValue(coeff, 'latex-sci'),
                FloatValue(cbd_avg, 1),
                FloatValue(cbd_max, 1))

    @cached_meth
    def pgd_acc(self):
        """a dict mapping from eps to pgd accuracy"""
        return self._load_pgd_acc('attack.json')

    @cached_meth
    def pgd_acc_hardtanh(self):
        return self._load_pgd_acc('attack-hardtanh.json')

    @cached_meth
    def _solver_stats_subset_ref(self):
        if self._name.startswith('mnist'):
            intersect = ('mnist-l-advnone-cbd0', '0.1')
        elif self._name.startswith('cifar10'):
            intersect = ('cifar10-l-advnone-cbd0', ADV2_255)
        else:
            raise RuntimeError('unknown expr')
        other, other_eps = intersect
        other = self._all_experiments[other]
        other_raw = other._all_solver_raw()
        return other_raw[f'eval-minisatcs-verify-{other_eps}.json']

    def _load_pgd_acc(self, fname):
        with (self._data_dir / fname).open() as fin:
            data = json.load(fin)
        assert data['pgd_steps'] == 100
        return {k: PercentValue(v, 2) for k, v in data['pgd'].items()}

    def _eval_cbd_stats(self):
        print(f'evaluating CBD losses for {self._data_dir} ...', end=' ',
              flush=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = (ModelHelper.
               create_with_load(self._data_dir / 'last.pth').
               to(device).
               eval())
        return eval_cbd_stats(self._args, net, device)

    def get_unused_solver_stat(self):
        """return a list of unaccessed solver stats"""

        ret = []
        for k, v in self._all_solver_raw().items():
            used_as_full = self._used_solver_stats.get(k)
            if used_as_full is None:
                ret.append(k)
            elif (not used_as_full and
                  len(v) > len(self._solver_stats_subset_ref())):
                ret.append(f'{k} (full)')
        return ret


class ValueRange:
    val = (float('inf'), float('-inf'))
    _sum = None
    _cnt = None

    def __init__(self, *, calc_avg=False):
        if calc_avg:
            self._sum = 0
            self._cnt = 0

    def update(self, *args):
        a, b = self.val
        for v in args:
            assert isinstance(v, float)
            a = min(a, v)
            b = max(b, v)
            if self._sum is not None:
                self._sum += a
                self._cnt += 1
        self.val = (a, b)

    @property
    def average(self):
        return self._sum / self._cnt

    def __repr__(self):
        return f'ValueRange{self.val}'


class Experiments:
    args = None
    out_dir: Path = None

    _experiments = None

    _used_experiments = None

    _mcs_speedup_range = None
    """mean speedup of minisatcs compared to other solvers"""

    _mcs_speedup_range_mid = None
    """median speedup of minisatcs compared to other solvers"""

    _latex_defs = None

    def __init__(self, args):
        self.args = args
        self._mcs_speedup_range = ValueRange(calc_avg=True)
        self._mcs_speedup_range_mid = ValueRange()
        p = Path(args.train_dir)
        experiments = {}
        for i in p.iterdir():
            if i.is_dir():
                expr = SingleExperiment(args, i, self)
                experiments[expr.name] = expr
        self._experiments = experiments
        self.out_dir = Path(args.out_dir)

        self._latex_defs = {}
        self._used_experiments = set()

    def __getitem__(self, key) -> SingleExperiment:
        self._used_experiments.add(key)
        return self._experiments[key]

    def _open_outfile(self, name):
        return (self.out_dir / name).open('w')

    def gen_table_ensemble(self):
        idx_row = [
            r'\texttt{conv-small}',
            r'\texttt{conv-large}',
            r'ensemble',
        ]
        idx_col = [
            'Test Accuracy',
            'Mean Solve Time (s)',
            'Attack Success Rate',
        ]
        data = np.empty((len(idx_row), len(idx_col)), dtype=object)

        exp = self[f'mnist-l-adv0.3-cbd{CHOSEN_CBD[0]}']
        exp_s = self[f'mnist-s-adv0.3']
        st = [
            exp_s.solver_stats('0.3'),
            exp.solver_stats('0.3'),
            exp.solver_stats('0.3', 'ensemble'),
        ]
        data[:, 0] = [
            exp_s.test_acc(),
            exp.test_acc(),
            exp.ensemble_test_acc('0.3'),
        ]
        data[:, 1] = [i.solve_time for i in st]
        data[:, 2] = [i.robust_prob.neg().set_precision(2) for i in st]

        df = pd.DataFrame(data, index=idx_row, columns=idx_col)
        with self._open_outfile('table-ensemble.tex') as fout:
            latex: str = df.to_latex(
                escape=False,
                column_format='lR{9ex}R{10ex}R{8ex}')
            fout.write(latex)
        with self._open_outfile('table-ensemble-wide.tex') as fout:
            latex: str = df.to_latex(
                escape=False,
                column_format='lrrr')
            fout.write(latex)

    def gen_fig_cmp_minisat(self):
        METHODS = ['Z3 (sub40)',
                   'MiniSat-seqcnt (sub40)', 'MiniSat-cardnet (sub40)',
                   'RoundingSat (sub40)', 'Ours: MiniSatCS (sub40)',
                   'Ours: MiniSatCS (full)',
                   'Narodytska et al. (sub100)']
        TIME_narodytska2020in = FloatValue(5.1, 3)
        EPS_DISP = ['20/255', '0.3']
        EPS_FILE = [ADV20_255, '0.3']

        fig_h, ax_h = plt.subplots(figsize=(8, 3))
        fig_v, ax_v = plt.subplots(figsize=(9, 6.5))

        solver_test_size = None
        exp = self['mnist-mlp']

        colors = ['#e41a1c', '#377eb8']

        NR_CMP = len(METHODS) - 1  # excluding Narodytska

        xmin_min = float('inf')
        for eps_i, eps in enumerate(EPS_FILE):
            xmean = []
            xmid = []
            xmin = []
            xmax = []
            for meth_i, meth in enumerate(
                    ['z3', 'm22', 'm22-cardnet', 'pb',
                     'minisatcs', 'minisatcs']):
                solver_st = exp.solver_stats(eps, meth,
                                             use_subset=(meth_i != NR_CMP-1))
                xmean.append(solver_st.solve_time.vnum)
                xmin.append(solver_st.solve_time_min.vnum)
                xmax.append(solver_st.solve_time_max.vnum)
                xmid.append(solver_st.solve_time_mid.vnum)

                if not self.args.add_unfinished and meth_i != NR_CMP-1:
                    if solver_test_size is None:
                        solver_test_size = solver_st.num
                    assert solver_test_size == solver_st.num, (
                        solver_test_size, solver_st.num)

            ycoord = np.arange(NR_CMP, dtype=np.float32)
            if eps_i == 0:
                ycoord -= 0.1
            else:
                ycoord += 0.1

            xmean, xmin, xmax = map(np.array, [xmean, xmin, xmax])

            range_kw = dict(marker='o', ls='', capsize=2, capthick=1,
                            color=colors[eps_i])
            mid_kw = dict(ls='', marker='^', color=colors[eps_i],
                          markersize=7)

            ax_h.errorbar(xmean, ycoord, xerr=[xmean - xmin, xmax - xmean],
                          **range_kw)
            ax_h.plot(xmid, ycoord, **mid_kw)
            ax_v.errorbar(ycoord, xmean, yerr=[xmean - xmin, xmax - xmean],
                          **range_kw)
            ax_v.plot(ycoord, xmid, **mid_kw)
            xmin_min = min(xmin_min, min(xmin))

        ax_h.plot([TIME_narodytska2020in.vnum], [NR_CMP],
                  marker='o', color=colors[0])
        ax_v.plot([NR_CMP], [TIME_narodytska2020in.vnum],
                  marker='o', color=colors[0])

        xlim = (xmin_min * 0.9, 1e4)
        ylim = (-0.49, 4.49)
        ax_h.set_xlabel('Time (seconds) in Log Scale')
        ax_h.set_xscale('log')
        ax_h.set_xlim(xlim)
        ax_h.set_yticks(np.arange(len(METHODS)))
        ax_h.set_yticklabels(METHODS)
        ax_h.set_ylim(ylim)
        ax_h.set_yticks(np.arange(len(METHODS)) + 0.5, minor=True)
        ax_h.grid(axis='y', which='minor')
        ax_h.grid(axis='x', which='major')

        ax_v.set_ylabel('Time (seconds) in Log Scale')
        ax_v.set_yscale('log')
        ax_v.set_ylim(xlim)
        ax_v.set_xticks(np.arange(len(METHODS)))
        ax_v.set_xticklabels(METHODS, rotation=45)
        ax_v.set_xlim(ylim)
        ax_v.set_xticks(np.arange(len(METHODS)) + 0.5, minor=True)
        ax_v.grid(axis='x', which='minor')
        ax_v.grid(axis='y', which='major')

        for ax in ax_h, ax_v:
            for c, e in zip(colors, EPS_DISP):
                ax.plot([], [], color=c, ls='-', label=rf'$\epsilon={e}$')
            ax.plot([], [], marker='o', ls='none', label='mean', color='black')
            ax.plot([], [], marker='^', ls='none', label='median',
                    color='black')
            ax.legend(loc='best', fancybox=True, framealpha=0.9,
                      borderpad=1, frameon=True)


        fig_h.tight_layout()
        fig_h.savefig(str(self.out_dir / 'fig-cmp-minisat.pdf'),
                    metadata={'CreationDate': None})

        fig_v.tight_layout()
        fig_v.savefig(str(self.out_dir / 'fig-cmp-minisat-vert.pdf'),
                    metadata={'CreationDate': None})


    @classmethod
    def _make_refdata_xiao(cls):
        npydict = lambda **kwargs: {k: np.array(v) for k, v in kwargs.items()}

        small = npydict(
            test_acc=[0.9868, 0.9733, 0.6112, 0.4045],
            pgd_acc=[0.9513, 0.9205, 0.4992, 0.2678],
            prov_acc=[0.9433, 0.8068, 0.4593, 0.2027],
            prov_upper=[0.9438, 0.8170, 0.4779, 0.2274],
            prov_time=[0.49, 2.78, 13.5, 22.33],
            build_time=[4.98, 4.34, 52.58, 38.34],
        )
        large = npydict(
            test_acc=[0.9895, 0.9754, 0.6141, 0.4281],
            pgd_acc=[0.9658, 0.9325, 0.5061, 0.2869],
            prov_acc=[0.9560, 0.5960, 0.4140, 0.1980],
            prov_upper=[0.9560, 0.8370, 0.5100, 0.2520],
            prov_time=[0.27, 37.45, 29.88, 20.14],
            build_time=[156.74, 166.39, 335.97, 401.72],
        )

        def fix(x):
            x['total_time'] = x['prov_time'] + x['build_time']
            x['timeout'] = x['prov_upper'] - x['prov_acc']
            return x

        return fix(small), fix(large)

    def gen_e2e_cmp_plot(self):
        xiao_small, xiao_large = self._make_refdata_xiao()

        fig, ax = plt.subplots()

        colors = COLOR4

        min_time = float('inf')
        max_time = float('-inf')
        def draw(time, acc, is_large, marker, color):
            nonlocal min_time, max_time
            min_time = min(time, min_time)
            max_time = max(time, max_time)
            kw = dict(edgecolors=color, linewidths=2)
            if is_large:
                kw['color'] = color
            else:
                kw['facecolors'] = 'none'
            ax.scatter([time], [acc * 100], marker=marker, **kw)

        for dset_i, dset in enumerate(['mnist', 'cifar10']):
            if dset_i == 0:
                eps_range = [('0.1',) * 3, ('0.3', ) * 3]
            else:
                eps_range = [(r'\frac{2}{255}', '2', ADV2_255),
                             (r'\frac{8}{255}', '8', ADV8_255)]
            for eps_i, (eps_disp, eps_file, eps_vrf) in enumerate(eps_range):
                idx = dset_i * 2 + eps_i
                color = colors[idx]

                draw(xiao_small['total_time'][idx],
                     xiao_small['prov_acc'][idx],
                     False, 's', color)
                draw(xiao_large['total_time'][idx],
                     xiao_large['prov_acc'][idx],
                     True, 's', color)

                st = self[f'{dset}-s-adv{eps_file}'].solver_stats(eps_vrf)
                draw(st.solve_time.vnum, st.robust_prob.vnum, False,
                     'o', color)
                st = (self[f'{dset}-l-adv{eps_file}-cbd{CHOSEN_CBD[dset_i]}'].
                      solver_stats(eps_vrf))
                draw(st.solve_time.vnum, st.robust_prob.vnum, True,
                     'o', color)

                ax.plot([], [], color=color,
                        label=fr'{dset.upper()} $\epsilon={eps_disp}$')

        def add_legend(marker, meth):
            ax.plot([], [], color='black', marker=marker, mfc='none', ls='none',
                    mew=2, label=f'{meth} small')
            ax.plot([], [], color='black', marker=marker, ls='none',
                    mew=2, label=f'{meth} large')

        add_legend('s', 'Xiao et al.')
        add_legend('o', 'EEV')

        ax.set_xlabel('Mean Verification Time (seconds) in Log Scale')
        ax.set_ylabel('Verifiable Accuracy (%)')
        ax.set_xscale('log')
        ax.set_xlim(min_time / 2, max_time * 50)
        ax.grid(which='major', axis='both')

        ax.legend(loc='upper right', fancybox=True, frameon=True,
                  framealpha=0.8, prop={'size': 'small'})

        fig.tight_layout()
        fig.savefig(str(self.out_dir / 'fig-e2e-cmp.pdf'),
                    metadata={'CreationDate': None})

    def gen_cmp_singledset(self):
        xiao_small, xiao_large = self._make_refdata_xiao()
        eev_se = self['cifar10-s-adv8']
        eev_le = self[f'cifar10-l-adv8-cbd{CHOSEN_CBD[1]}']
        eev_s = eev_se.solver_stats(ADV8_255)
        eev_l = eev_le.solver_stats(ADV8_255)

        x_labels = [
            'BNN small', 'BNN large',
            'Real-valued small', 'Real-valued large'
        ]
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))

        for i in ax0, ax1:
            i.set_xticks(np.arange(len(x_labels)))
            i.set_xticklabels(x_labels, rotation=45)

        xs = np.arange(len(x_labels))
        def make_y(exeev, xkey, k=1, use_solver=True):
            if use_solver:
                eev = [eev_s, eev_l]
            else:
                eev = [eev_se, eev_le]
            return [i * k for i in (
                exeev(eev[0]), exeev(eev[1]),
                xiao_small[xkey][3], xiao_large[xkey][3])]


        ys = make_y(lambda x: x.solve_time.vnum, 'prov_time')
        ax0.scatter(xs[:2], ys[:2], marker='*', color='green', s=130)
        ax0.scatter(xs[2:], ys[2:], color='black', s=50)
        ax0.set_ylabel('Mean Solve Time (seconds) in Log Scale')
        ax0.set_yscale('log')
        ax0.set_ylim(min(ys) / 4, max(ys) * 4)
        ax0.grid(which='major', axis='y')

        colors = ['#ffeda0', '#feb24c', '#f03b20']
        ys0 = make_y(lambda x: x.robust_prob.vnum, 'prov_acc', 100)
        ys1 = make_y(lambda x: x.solve_prob.neg().vnum, 'timeout', 100)
        ys2 = make_y(lambda x: x.test_acc().vnum, 'test_acc', 100,
                     use_solver=False)
        ax1.bar(xs, ys2, label='Natural Test Accuracy', width=0.6,
                color=colors[0])
        ax1.bar(xs, ys0, label='Verfiable Accuracy', width=0.6,
                color=colors[1])
        ax1.bar(xs, ys1, bottom=ys0, label='Timeout', width=0.6,
                color=colors[2])
        ax1.set_ylabel('Accuracy (%)')
        ax1.grid(which='major', axis='y')
        ax1.legend(loc='upper left', fancybox=True, framealpha=0.9,
                   borderpad=1, frameon=True)
        ax1.set_ylim(0, 55)

        fig.suptitle(r'Comparing Verification Performance on CIFAR10 '
                     r'with $\epsilon=\frac{8}{255}$ Timeout=120s')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        for ext in ['png', 'pdf']:
            fig.savefig(str(self.out_dir / f'fig-cmp-singledset.{ext}'),
                        metadata={'CreationDate': None})

    def gen_table_summary(self):
        xiao_data_small, xiao_data_large = self._make_refdata_xiao()

        idx_row = list(itertools.product(
            [
                r'MNIST $\epsilon=0.1$',
                r'MNIST $\epsilon=0.3$',
                r'CIFAR10 $\epsilon=\frac{2}{255}$',
                r'CIFAR10 $\epsilon=\frac{8}{255}$'
            ],
            [
                r'EEV S',
                r'EEV L',
                r'\citet{xiao2018training} S',
                r'\citet{xiao2018training} L\tnote{*}',
            ]
        ))
        idx_col = (
            list(itertools.product(
                ['Mean Time (s)'],
                ['Build', 'Solve', 'Total'])) +
            list(itertools.product(
                ['Accuracy'],
                ['Verifiable', 'Natural', 'PGD']
            )) +
            [('Timeout', '~')]
        )
        data = np.empty((len(idx_row), len(idx_col)), dtype=object)

        def fill_existing(rows, source):
            mkacc = lambda x: [PercentValue(i, 2) for i in x]
            mktime = lambda x: [FloatValue(i, 2) for i in x]
            data[rows, :] = np.array([
                mktime(source['build_time']),
                mktime(source['prov_time']),
                mktime(source['total_time']),
                mkacc(pacc := source['prov_acc']),
                mkacc(source['test_acc']),
                mkacc(source['pgd_acc']),
                mkacc(source['prov_upper'] - pacc),
            ], dtype=object).T

        fill_existing(np.arange(2, 16, 4), xiao_data_small)
        fill_existing(np.arange(3, 16, 4), xiao_data_large)

        TRAIN_EPS = {
            'mnist': ['0.1', '0.3'],
            'cifar10': ['2', '8'],
        }

        for dset_i, dset in enumerate(['mnist', 'cifar10']):
            for model_i, model in enumerate('sl'):
                for train_eps_i, train_eps in enumerate(TRAIN_EPS[dset]):
                    rbase = dset_i * 8 + train_eps_i * 4 + model_i
                    suffix = ''
                    if model_i == 1:
                        suffix = f'-cbd{CHOSEN_CBD[dset_i]}'
                    exp = self[f'{dset}-{model}-adv{train_eps}{suffix}']

                    if dset_i == 0:
                        test_eps = train_eps
                    else:
                        test_eps = [ADV2_255, ADV8_255][train_eps_i]
                    solver_st = exp.solver_stats(test_eps)
                    data[rbase, :] = [
                        solver_st.build_time.set_precision(4),
                        solver_st.solve_time.set_precision(4),
                        (solver_st.build_time +
                         solver_st.solve_time).set_precision(4),
                        solver_st.robust_prob,
                        exp.test_acc(),
                        exp.pgd_acc()[test_eps],
                        solver_st.solve_prob.neg()
                    ]


        for c, name in zip([1, 2], ['solve', 'total']):
            r = ValueRange()
            r.update(*(data[2::4, c] / data[0::4, c]))
            r.update(*(data[3::4, c] / data[1::4, c]))

            self._latex_defs.update([
                (f'{name}SpeedupMin', fr'{r.val[0]:.2f}'),
                (f'{name}SpeedupMax', fr'{r.val[1]:.2f}'),
            ])

        for i in data[:, -1]:
            i.set_empty_for_zero()

        for r in range(4):
            for i in data[3+r*4, [3, -1]]:
                i.set_precision(1)

        df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(idx_row),
                          columns=pd.MultiIndex.from_tuples(idx_col))
        latex: str = df.to_latex(
            escape=False, multirow=True,
            column_format='ll' + 'r' * data.shape[1], multicolumn_format='c')
        latex = (add_second_row_title(latex, '').
                 replace('{4}{*}', '{4}{4.5em}')    # multirow
                 )
        latex = fix_cline(latex)
        with self._open_outfile('table-summary.tex') as fout:
            fout.write(latex)

    def gen_table_cmp_acc_multi_eps(self):
        idx_row = list(itertools.product(
            [
                r'MNIST $\epsilon=0.1$',
                r'MNIST $\epsilon=0.3$',
                r'CIFAR10 $\epsilon=2/255$',
                r'CIFAR10 $\epsilon=8/255$'
            ],
            [
                r'\texttt{conv-small}',
                r'\texttt{conv-large}',
            ]
        ))
        idx_col = (
            [('Test Accuracy', '~')] +
            list(itertools.product(
                ['PGD Adversarial Accuracy',
                 'Verifiable Accuracy',
                 'Mean Solve Time',
                 'Solver Timeout',
                 'Mean Build+Solve Time',
                 ],
                [rf'$\epsilon=\epsilon_{i}$' for i in range(3)],
            )))
        idx_col.insert(7, ('Sparsity', '~'))
        data = np.empty((len(idx_row), len(idx_col)), dtype=object)

        TRAIN_EPS = {
            'mnist': ['0.1', '0.3'],
            'cifar10': ['2', '8'],
        }
        TEST_EPS = {
            'mnist': ['0.1', '0.2', '0.3'],
            'cifar10': [ADV2_255, ADV5_255, ADV8_255]
        }

        for dset_i, dset in enumerate(['mnist', 'cifar10']):
            for model_i, model in enumerate('sl'):
                for train_eps_i, train_eps in enumerate(TRAIN_EPS[dset]):
                    rbase = dset_i * 4 + train_eps_i * 2 + model_i
                    suffix = ''
                    if model_i == 1:
                        suffix = f'-cbd{CHOSEN_CBD[dset_i]}'
                    exp = self[f'{dset}-{model}-adv{train_eps}{suffix}']

                    pgd_acc = exp.pgd_acc()
                    data[rbase, 0] = exp.test_acc()
                    data[rbase, 7] = exp.sparsity().tot_avg
                    for test_eps_i, test_eps in enumerate(TEST_EPS[dset]):
                        data[rbase, test_eps_i + 1] = pgd_acc[test_eps]
                        solver_st = exp.solver_stats(test_eps)
                        data[rbase, test_eps_i + 4] = (
                            solver_st.robust_prob
                        )
                        data[rbase, test_eps_i + 8] = (
                            solver_st.solve_time.set_precision(4)
                        )
                        data[rbase, test_eps_i + 11] = (
                            solver_st.solve_prob.neg().set_empty_for_zero()
                        )
                        data[rbase, test_eps_i + 14] = (
                            solver_st.build_time.set_precision(4) +
                            solver_st.solve_time
                        )

        def make_table(data, idx_col, out_name):
            idx_col = pd.MultiIndex.from_tuples(idx_col)
            df = pd.DataFrame(data, index=idx_row, columns=idx_col)
            latex: str = df.to_latex(
                escape=False, multirow=True,
                column_format='ll' + 'r'*data.shape[1],
                multicolumn_format='c')
            # width of multirow
            latex = latex.replace('{*}', '{5em}')

            # add title
            lines = latex.split('\n')
            def replace_head(l, a, b):
                head = lines[l].split(' & ')
                assert head[0].isspace() and head[1].isspace(), head[:2]
                head[0] = a
                head[1] = b
                lines[l] = ' & '.join(head)
            replace_head(2, 'Dataset', 'Network')

            latex = '\n'.join(lines)
            latex = fix_cline(latex)
            latex = add_second_row_title(latex, r'Training $\epsilon$')

            with self._open_outfile(out_name) as fout:
                fout.write(latex)

        idx_row = pd.MultiIndex.from_tuples(idx_row)
        idx_col = pd.MultiIndex.from_tuples(idx_col)
        sep = 8
        make_table(data[:, :sep], idx_col[:sep], 'table-multi-eps-acc.tex')
        make_table(data[:, sep:], idx_col[sep:], 'table-multi-eps-time.tex')

    def gen_table_cmp_mnist_mlp(self):
        idx_row = [
            'Test Accuracy',
            r'\#Non-zero Params',
        ]
        idx_col = [
            'MNIST-MLP', r'\cite{narodytska2020in}'
        ]
        fmt_k = r'${}\mathrm{{K}}$'.format
        data = np.empty((2, 2), dtype=object)
        exp = self['mnist-mlp']
        data[:, 0] = [
            exp.test_acc(),
            fmt_k(int(exp.sparsity().nr_non_zero / 1000)),
        ]
        data[:, 1] = [
            r'95.2\%',
            fmt_k(20),
        ]
        df = pd.DataFrame(data, index=idx_row, columns=idx_col)
        with self._open_outfile('table-cmp-mnist-mlp.tex') as fout:
            fout.write(df.to_latex(escape=False, column_format='lrr'))

        self._latex_defs.update([
            ('mnistMlpPrec', data[0, 0]),
            ('mnistMlpNzp', data[1, 0]),
            ('mnistMlpRefPrec', data[0, 1]),
            ('mnistMlpRefNzp', data[1, 1]),
        ])

    def gen_table_cmp_binmask(self):
        idx_row_wide = [
            'Total Sparsity',
            'Layer-wise Sparsity',
            'Mean Solve Time (sub40)',
            'Max Solve Time (sub40)',
            'Natural Test Accuracy',
        ]
        idx_row = pd.MultiIndex.from_product([
            idx_row_wide,
            ['Ternary', 'BinMask']
        ])
        idx_col = [r'MNIST \hspace{1em} $\epsilon=0.1$',
                   r'CIFAR10 \hspace{1em} $\epsilon=2/255$']
        data = np.empty((len(idx_row), len(idx_col)), dtype=object)

        solver_test_size = None
        for dset_i, dset in enumerate(['mnist-s-advnone', 'cifar10-s-advnone']):
            for meth_i, meth in enumerate(['ternweight', 'binmask']):
                if meth_i == 0:
                    exp = self[f'{dset}-{meth}']
                else:
                    exp = self[dset]
                sp_layer, sp_tot, _ = exp.sparsity()
                eps = ['0.1', ADV2_255][dset_i]
                #eps = '0.01'
                solver_st = exp.solver_stats(eps, use_subset=True)
                data[meth_i::2, dset_i] = [
                    sp_tot,
                    ' '.join(map(str, sp_layer)),
                    solver_st.solve_time,
                    solver_st.solve_time_max,
                    exp.test_acc(),
                ]

                if not self.args.add_unfinished:
                    if solver_test_size is None:
                        solver_test_size = solver_st.num
                    assert solver_test_size == solver_st.num, (
                        solver_test_size, solver_st.num, dset, meth)

        df = pd.DataFrame(data, index=idx_row, columns=idx_col)
        latex = (df.to_latex(escape=False,
                            multirow=True, column_format='llrr').
                 replace('{*}', '{4.8em}'))
        latex = fix_cline(latex)
        with self._open_outfile(f'table-cmp-binmask-thin.tex') as fout:
            fout.write(latex)

        # wide format
        data_tern = data[::2, :, np.newaxis]
        data_bin = data[1::2, :, np.newaxis]
        data_wide = np.concatenate([data_tern, data_bin], axis=2)
        data_wide = data_wide.reshape(data.shape[0] // 2, data.shape[1] * 2)
        df = pd.DataFrame(data_wide, index=idx_row_wide,
                          columns=pd.MultiIndex.from_product([
                              idx_col, ['Ternary', 'BinMask']
                          ]))
        latex = df.to_latex(escape=False, multicolumn_format='c',
                             column_format='l' + 'r'*4)
        latex = add_second_row_title(latex, 'Sparsifier')
        with self._open_outfile(f'table-cmp-binmask-wide.tex') as fout:
            fout.write(latex)

    def gen_table_cmp_cbd(self):
        idx_row = [
            'Mean / Max Card Bound',
            'Mean Solve Time (sub40)',
            'Max Solve Time (sub40)',
            'Verifiable Accuracy',
            'Natural Test Accuracy',
            'PGD Accuracy',
            'First Layer / Total Sparsity',
        ]
        idx_col = ([[r'MNIST \hspace{1em} $\epsilon=0.3$', 'x']
                    for _ in range(4)] +
                   [[r'CIFAR10 \hspace{1em} $\epsilon=8/255$', 'x']
                    for _ in range(4)])
        data = np.empty((len(idx_row), len(idx_col)), dtype=object)

        solver_test_size = None
        for dset_i, dset in enumerate(['mnist-l-adv0.3', 'cifar10-l-adv8']):
            for meth_i in range(4):
                meth = f'cbd{meth_i}'
                exp = self[f'{dset}-{meth}']
                cbd_coeff, cbd_avg, cbd_max = exp.cbd_stats()
                eps = ['0.3', ADV8_255][dset_i]
                solver_st = exp.solver_stats(
                    eps,
                    name=('minisatcs-sub' if dset_i == 0 and meth_i == 2
                          else None),
                    use_subset=True)
                solver_st_full = exp.solver_stats(eps)
                if solver_st_full.num > solver_st.num:
                    acc_full = solver_st_full.robust_prob
                    if not self.args.add_unfinished:
                        assert solver_st_full.solve_prob.vnum == 1, (
                            dset, eps, meth
                        )
                else:
                    acc_full = '-'
                cidx = dset_i * 4 + meth_i
                idx_col[cidx][1] = f'${cbd_coeff}$'
                sparsity = exp.sparsity()
                data[:, cidx] = [
                    f'{cbd_avg}~/~{cbd_max}',
                    solver_st.solve_time,
                    solver_st.solve_time_max,
                    acc_full,
                    exp.test_acc(),
                    exp.pgd_acc()[eps],
                    f'{sparsity.layerwise[0]}~/~{sparsity.tot_avg}',
                ]
                if not self.args.add_unfinished:
                    if solver_test_size is None:
                        solver_test_size = solver_st.num
                    assert solver_test_size == solver_st.num, (
                        solver_test_size, solver_st.num, dset, meth)

        df = pd.DataFrame(data, index=idx_row,
                          columns=pd.MultiIndex.from_tuples(idx_col))
        with self._open_outfile('table-cmp-cbd.tex') as fout:
            latex: str = df.to_latex(
                escape=False, column_format='l' + 'r'*8,
                multicolumn_format='c',
            )
            latex = add_second_row_title(latex, r'CBD Loss Penalty ($\eta$)')
            fout.write(latex)

    def gen_table_method_cmp(self):
        self._gen_table_method_cmp_dset_part('mnist', '0.3', '0.3')
        self._gen_table_method_cmp_dset_part('cifar10', '8', ADV8_255)

        get_exp = lambda e: self['mnist-l-adv' + e]
        cbd = lambda e: get_exp(e).cbd_stats()[1]
        cbd_max = lambda e: get_exp(e).cbd_stats()[2]
        sparsity = lambda e: r'\;'.join(map(str,
                                            get_exp(e).sparsity().layerwise))

        f2 = '{:.2f}'.format

        r = self._mcs_speedup_range
        rmid = self._mcs_speedup_range_mid
        self._latex_defs.update([
            ('mcsSpeedupMin', f2(r.val[0])),
            ('mcsSpeedupAvg', f2(r.average)),
            ('mcsSpeedupMax', f2(r.val[1])),
            ('mcsSpeedupMidMin', f2(rmid.val[0])),
            ('mcsSpeedupMidMax', f2(rmid.val[1])),
            ('ternWeightCardBound', cbd('none-ternweight')),
            ('ternWeightCbdCardBound', cbd('none-ternweight-cbd')),
            ('ternWeightCbdCardBoundMax', cbd_max('none-ternweight-cbd')),
            ('ternWeightAdvCbdCardBoundMax', cbd_max('0.3-ternweight-cbd')),
            ('ternWeightStrongCbdCardBound', cbd('none-ternweight-cbd1')),
            ('ternWeightStrongCbdCardBoundMax', cbd_max('none-ternweight-cbd1')),
            ('ternWeightBinMaskCbdCardBound', cbd('none-cbd1')),
            ('ternWeightBinMaskCbdCardBoundMax', cbd_max('none-cbd1')),

            ('layerSparsityTernary', sparsity('none-ternweight')),
            ('layerSparsityBinMask', sparsity('none-cbd0')),
            ('layerSparsityTernaryCardBound', cbd('none-ternweight')),
            ('layerSparsityBinMaskCardBound', cbd('none-cbd0')),
        ])

    def _gen_table_method_cmp_dset_part(self, dset, eps_train, eps_test):
        idx_row, data_num = self._gen_table_method_cmp_dset_data(
            dset, eps_train, eps_test)

        idx_col = [
            'Mean Solve Time', 'Median Solve Time',
            'Timeout', 'Verifiable Accuracy',
        ]
        col_formats = [
            'l', 'L{14ex}', 'L{17ex}',
            'R{11ex}', 'R{9ex}',
            'l', 'R{12ex}',  'R{11ex}', 'r', 'R{12ex}',
        ]


        data = np.empty((data_num.shape[0], 4), dtype=np.object)
        for i in range(data_num.shape[0]):
            eps, net, meth, solver = idx_row[i]
            idx_row[i] = (eps, net, meth,
                          str(data_num[i][0]), str(data_num[i][1]),
                          solver)
            data[i] = data_num[i, 2:]

        df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(idx_row),
                          columns=idx_col)
        with self._open_outfile(f'table-method-cmp-{dset}-part.tex') as fout:
            latex: str = df.to_latex(
                escape=False, multirow=True,
                column_format=''.join(col_formats)
            )
            latex = latex.replace(
                r'\multirow{3}{*}{BinMask+CBD}',
                r'\multirow{3}{7ex}{BinMask +CBD}')

            lines = latex.split('\n')
            head = lines[2].split(' & ')
            head[:6] = [
                r'$\epsilon_{\text{train}}$',
                'Network Architecture',
                'Training Method',
                'Test Accuracy',
                'Sparsity',
                'Solver',
            ]
            lines[2] = ' & '.join(head)
            latex = '\n'.join(lines)

            latex = fix_cline(latex)

            fout.write(latex)

    def _gen_table_method_cmp_dset_data(self, dset, eps_train, eps_test):
        solver = 'MiniSatCS'
        solver_ref_m22 = 'MiniSat'
        solver_ref_z3 = 'Z3'
        solver_ref_roundingsat = 'RoundingSat'
        solver_ref_m22_cn = 'MiniSat-CN'
        meth_map = {
            'Ternary': lambda eps: 'ternweight',
            'Ternary (wd0)': lambda eps: 'ternweight-wd0',
            'Ternary (wd1)': lambda eps: 'ternweight-wd1',
            'Ternary+CBD': lambda eps: 'ternweight-cbd',
            'Ternary+10xCBD': lambda eps: 'ternweight-cbd1',
            'BinMask': lambda eps: 'cbd0',
            'BinMask+CBD': lambda eps: (
                'cbd1' if eps == '0' else
                ('cbd3' if dset == 'mnist' else 'cbd2'))
        }

        net_name2abbr = lambda s, *, l=len(r'\texttt{conv-'): s[l]
        def eps_full_info(e: str):
            if e == '0':
                return '0', 'none', ('0.1' if dset == 'mnist' else ADV2_255)
            assert e == eps_train
            disp = e
            if e.isdigit():
                disp = f'${e}/255$'
            return disp, e, eps_test

        desc_eps0 = [
            (r'\texttt{conv-small}',
             [
                 ['Ternary', [solver]],
                 ['BinMask', [solver, solver_ref_m22, solver_ref_z3,
                              solver_ref_roundingsat]],
             ]),
            (r'\texttt{conv-large}',
             [
                 ['Ternary', [solver]],
                 ['BinMask', [solver]],
                 ['BinMask+CBD', [solver, solver_ref_m22, solver_ref_z3,
                                  solver_ref_roundingsat]],
             ]),
        ]
        desc_eps1 = copy.deepcopy(desc_eps0)
        if dset == 'mnist':
            desc_eps1[0][1][1][1].append(solver_ref_m22_cn)
            desc_eps1[1][1][2][1].append(solver_ref_m22_cn)

            del desc_eps1[0][1][0]  # del ternweight, add wd0 and wd1
            desc_eps1[0][1].insert(0, ['Ternary (wd0)', [solver]])
            desc_eps1[0][1].insert(1, ['Ternary (wd1)', [solver]])
            desc_eps0[1][1].insert(1, ['Ternary+CBD', [solver]])
            desc_eps0[1][1].insert(2, ['Ternary+10xCBD', [solver]])
            desc_eps1[1][1].insert(1, ['Ternary+CBD',
                                       [solver, solver_ref_m22, solver_ref_z3,
                                        solver_ref_roundingsat]])
        if dset == 'cifar10':
            desc_eps0[0][1][0][1].extend([solver_ref_m22, solver_ref_z3,
                                          solver_ref_roundingsat])
            for i in range(2):
                desc_eps1[1][1][i][1].append(solver_ref_roundingsat)

        idx_row = gen_prod_treelike([
            ('0', desc_eps0),
            (eps_train, desc_eps1)
        ])

        data = np.empty((len(idx_row), 6), dtype=object)
        solver_test_size = None
        for row_id, row_desc in enumerate(idx_row):
            (eps, net, meth, slv) = row_desc
            eps_disp, case_expr_eps, case_test_eps = eps_full_info(eps)
            row_desc[0] = eps_disp
            net = net_name2abbr(net)
            expr_name = expr_base_name = f'{dset}-{net}-adv{case_expr_eps}'
            if not (net == 's' and meth == 'BinMask'):
                expr_name += f'-{meth_map[meth](eps)}'
            expr = self[expr_name]
            name = None
            if slv == solver_ref_z3:
                name = 'z3'
            elif slv == solver_ref_m22:
                name = 'm22'
            elif slv == solver_ref_m22_cn:
                name = 'm22-cardnet'
            elif slv == solver_ref_roundingsat:
                name = 'pb'
            elif (slv, eps, net, meth) == (solver, '0', 'l', 'BinMask+CBD'):
                name = 'minisatcs-sub'
            solver_st = expr.solver_stats(case_test_eps, name, use_subset=True)
            data[row_id] = [
                expr.test_acc(),
                expr.sparsity().tot_avg,
                solver_st.solve_time,
                solver_st.solve_time_mid,
                solver_st.solve_prob.neg().set_empty_for_zero(),
                solver_st.robust_prob,
            ]
            if not self.args.add_unfinished:
                if solver_test_size is None:
                    solver_test_size = solver_st.num
                assert solver_test_size == solver_st.num, (
                    solver_test_size, solver_st.num, dset, meth)

        # update speedup statistics
        solver_cmp = [solver_ref_m22, solver_ref_z3, solver_ref_m22_cn,
                      solver_ref_roundingsat]
        for i, r0 in enumerate(idx_row):
            if r0[-1] == solver:
                t0 = data[i, 2].vnum
                t1 = data[i, 3].vnum
                j = i + 1
                while j < len(idx_row) and idx_row[j][-1] in solver_cmp:
                    assert idx_row[j][:-1] == r0[:-1]
                    self._mcs_speedup_range.update(data[j, 2].vnum / t0)
                    self._mcs_speedup_range_mid.update(data[j, 3].vnum / t1)
                    j += 1

        return idx_row, data

    def gen_hardtanh_cmp(self):
        exp_pgd = self['mnist-s-adv0.3-hardtanh']
        self._latex_defs['hardtanhCmpBaselinePGD'] = (
            exp_pgd.pgd_acc_hardtanh()['0.3'].neg()
        )
        self._latex_defs['hardtanhCmpBaselinePGDWithTanh'] = (
            exp_pgd.pgd_acc()['0.3'].neg()
        )

        exps = list(map(self.__getitem__,
                        ['cifar10-l-adv8-cbd3-hardtanh',
                         'cifar10-l-adv8-cbd3-tanh-noscale',
                         'cifar10-l-adv8-cbd3',
                         'cifar10-l-adv8-cbd3-vrf']))

        idx_row = ['Natural Test Accuracy', 'Verifiable Accuracy']
        idx_col = ['hard tanh', 'tanh', 'adaptive', 'adaptive + verifier adv']
        data = np.empty((len(idx_row), len(idx_col)), dtype=object)
        data[0] = [i.test_acc() for i in exps]
        data[1] = [i.solver_stats(ADV8_255).robust_prob for i in exps]
        df = pd.DataFrame(data, index=idx_row, columns=idx_col)
        latex: str = df.to_latex(escape=False, column_format='lrrrr')
        with self._open_outfile('table-cmp-tanh.tex') as fout:
            fout.write(latex)

    def gen_cmp_last_layer_bn(self):
        exp0 = self['cifar10-s-advnone']
        exp1 = self['cifar10-s-advnone-full-last-bn']
        exp2 = self['cifar10-s-advnone-no-last-bn']
        self._latex_defs.update({
            'cmpLastBnOursAcc': exp0.test_acc(),
            'cmpLastBnOursLoss': exp0.test_loss(),
            'cmpLastBnFullAcc': exp1.test_acc(),
            'cmpLastBnFullLoss': exp1.test_loss(),
            'cmpLastBnNoneAcc': exp2.test_acc(),
            'cmpLastBnNoneLoss': exp2.test_loss(),
        })

    def write_latex_defs(self):
        with self._open_outfile('numdefs.tex') as fout:
            for k, v in self._latex_defs.items():
                print(r'\newcommand{\%s}{%s}' % (k, v), file=fout)

    def print_unused_experiments(self):
        print('======= unused experiments =======')
        for k, v in sorted(self._experiments.items()):
            if k not in self._used_experiments:
                print(f'{k}')
            else:
                st = v.get_unused_solver_stat()
                if st:
                    print(f'solver stat in {k}: {st}')


def set_plot_style():
    # see http://www.jesshamrick.com/2016/04/13/reproducible-plots/
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    matplotlib.rcParams.update({
        'font.size': 10,
        'legend.fontsize': 'medium',
        'axes.labelsize': 'large',
        'xtick.labelsize': 'large',
        'ytick.labelsize': 'large',
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', help='training output dir')
    parser.add_argument('out_dir', help='latex data output dir')
    parser.add_argument('--add-unfinished', action='store_true',
                        help='add unfinished results')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument(
        '--data', default=default_dataset_root(),
        help='dir for training data')
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    set_plot_style()

    exp = Experiments(args)
    for i in dir(exp):
        if i.startswith('gen_'):
            print(f'executing {i}() ...')
            getattr(exp, i)()
    exp.write_latex_defs()
    print(f'all tested cases: {g_possible_tested_cases}')
    if not args.add_unfinished:
        assert len(g_possible_tested_cases) == 2

    exp.print_unused_experiments()

if __name__ == '__main__':
    main()
