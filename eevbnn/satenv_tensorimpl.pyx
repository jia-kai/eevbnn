# -*- coding: utf-8 -*-
# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython

cdef class TensorMathImpl:
    @classmethod
    def _canonize_weights(cls, weight, bias, sign):
        w = np.ascontiguousarray(weight, dtype=np.int32)
        b = np.ascontiguousarray(bias, dtype=np.int32)
        s = np.ascontiguousarray(sign, dtype=np.int32)
        assert np.all(w == weight)
        assert np.all(b == bias)
        assert np.all(s == sign) and set(s).issubset({-1, 1})
        return w, b, s

    @classmethod
    def conv2d(cls, self, name, weight, bias, sign, stride, padding):
        cdef int N, IC, IH, IW, OC, KH, KW, OH, OW
        cdef int n, oc, oh, ow, ic, kh, kw, ih, iw
        cdef int strd_h, strd_w, pad_h, pad_w, w
        cdef list acc
        assert (self.ndim == 4 and weight.ndim == 4
                and self.shape[1] == weight.shape[1]), (
                    f'conv shape mismatch: {self.shape}, {weight.shape}')
        strd_h, strd_w = stride
        pad_h, pad_w = padding
        N, IC, IH, IW = self.shape
        OC, _, KH, KW = weight.shape
        OH = (IH + pad_h * 2 - KH) // strd_h + 1
        OW = (IW + pad_w * 2 - KW) // strd_w + 1
        assert bias.shape == sign.shape == (OC, ), f'{bias.shape} {sign.shape}'
        result = np.empty((N, OC, OH, OW), dtype=object)
        weight, bias, sign = cls._canonize_weights(weight, bias, sign)

        cdef np.ndarray[object, ndim=4] npy_value = self.value
        cdef np.ndarray[object, ndim=4] npy_result = result
        cdef np.ndarray[np.int32_t, ndim=4] npy_w = weight
        cdef np.ndarray[np.int32_t, ndim=1] npy_b = bias
        cdef np.ndarray[np.int32_t, ndim=1] npy_s = sign

        for n in range(N):
            for oc in range(OC):
                for oh in range(OH):
                    for ow in range(OW):
                        acc = []
                        for ic in range(IC):
                            for kh in range(KH):
                                for kw in range(KW):
                                    ih = oh * strd_h + kh - pad_h
                                    iw = ow * strd_w + kw - pad_w
                                    if (ih >= 0 and ih < IH and
                                            iw >= 0 and iw < IW):
                                        w = npy_w[oc, ic, kh, kw]
                                        if w:
                                            acc.append(
                                                (npy_value[n, ic, ih, iw], w))
                        npy_result[n, oc, oh, ow] = self._dot_thresh(
                            f'{name}_{n}_{oc}_{oh}_{ow}',
                            acc, int(bias[oc]), int(sign[oc]))

        return self.make_replaced_value(result)

    @classmethod
    def conv2d_quant(cls, self, name, weight, bias, sign, stride, padding):
        from .satenv import BoolTensor

        cdef int N, IC, IH, IW, OC, KH, KW, OH, OW
        cdef int n, oc, oh, ow, ic, kh, kw, ih, iw
        cdef int strd_h, strd_w, pad_h, pad_w, w
        cdef list acc
        cdef list cur_value
        cdef int acc_offset
        assert (self.ndim == 4 and weight.ndim == 4
                and self.shape[1] == weight.shape[1]), (
                    f'conv shape mismatch: {self.shape}, {weight.shape}')
        strd_h, strd_w = stride
        pad_h, pad_w = padding
        N, IC, IH, IW = self.shape
        OC, _, KH, KW = weight.shape
        OH = (IH + pad_h * 2 - KH) // strd_h + 1
        OW = (IW + pad_w * 2 - KW) // strd_w + 1
        assert bias.shape == sign.shape == (OC, ), f'{bias.shape} {sign.shape}'
        result = BoolTensor(self.env, np.empty((N, OC, OH, OW), dtype=object))
        weight, bias, sign = cls._canonize_weights(weight, bias, sign)

        cdef np.ndarray[object, ndim=4] npy_value = self.value
        cdef np.ndarray[np.int32_t, ndim=4] npy_offset = self.offset
        cdef np.ndarray[object, ndim=4] npy_result = result.value
        cdef np.ndarray[np.int32_t, ndim=4] npy_w = weight
        cdef np.ndarray[np.int32_t, ndim=1] npy_b = bias
        cdef np.ndarray[np.int32_t, ndim=1] npy_s = sign

        for n in range(N):
            for oc in range(OC):
                for oh in range(OH):
                    for ow in range(OW):
                        acc = []
                        acc_offset = 0
                        for ic in range(IC):
                            for kh in range(KH):
                                for kw in range(KW):
                                    ih = oh * strd_h + kh - pad_h
                                    iw = ow * strd_w + kw - pad_w
                                    if (ih >= 0 and ih < IH and
                                            iw >= 0 and iw < IW):
                                        w = npy_w[oc, ic, kh, kw]
                                        if w:
                                            cur_value = npy_value[n, ic, ih, iw]
                                            for i in cur_value:
                                                acc.append((i, w))
                                            acc_offset += (
                                                npy_offset[n, ic, ih, iw] * w)
                        npy_result[n, oc, oh, ow] = result._dot_thresh(
                            f'{name}_{n}_{oc}_{oh}_{ow}',
                            acc, int(npy_b[oc] + acc_offset), int(npy_s[oc]))

        return result

    @classmethod
    def matmul(cls, self, name, weight, bias, sign):
        cdef int M, K, N, i, j, k, w
        cdef list acc

        assert (self.ndim == 2 and weight.ndim == 2 and
                self.shape[1] == weight.shape[0]), (
                    f'matmul shape mismatch: {self.shape} {weight.shape}')
        M, K = self.shape
        _, N = weight.shape
        assert bias.shape == sign.shape == (N, ), f'{bias.shape} {sign.shape}'
        result = np.empty((M, N), dtype=object)

        weight, bias, sign = self._canonize_weights(weight, bias, sign)

        cdef np.ndarray[object, ndim=2] npy_value = self.value
        cdef np.ndarray[object, ndim=2] npy_result = result

        for i in range(M):
            for j in range(N):
                acc = []
                for k in range(K):
                    w = weight[k, j]
                    if w:
                        acc.append((npy_value[i, k], w))
                npy_result[i, j] = self._dot_thresh(
                    f'{name}_{i}_{j}', acc, int(bias[j]), int(sign[j]))
        return self.make_replaced_value(result)
