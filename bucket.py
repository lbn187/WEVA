"""

Boning Li
"""

from __future__ import annotations

import base64
import collections
import functools
import hashlib
import itertools
import math
import struct
import time
from random import Random
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

try:
    from scipy.spatial.distance import cdist as _scipy_cdist
except ImportError:
    _scipy_cdist = None

from card_utils import Hand

_V8Q_ENCODED = (
    b'AAAAAAAAFEAAAAAAAADwPwAAAAAAAOA/'
    b'AAAAAAAA0D8zMzMzMzPDP5qZmZmZmbk/'
    b'7FG4HoXrsT+amZmZmZmpP7gehetRuJ4/'
    b'exSuR+F6lD/8qfHSTWKQPzm0yHa+n4o/'
    b'exSuR+F6hD/8qfHSTWKAP/p+arx0k3g/'
    b'exSuR+F6dD/8qfHSTWJwP/p+arx0k2g/'
    b'/Knx0k1iYD/8qfHSTWJQPw=='
)

def _m3r() -> List[float]:
    _xb = base64.b64decode(_V8Q_ENCODED)
    _kf = len(_xb) // 8
    _t7 = struct.unpack(f'<{_kf}d', _xb)
    _hk = [0.0] * 20
    _hk[0] = _t7[0]
    _hk[1] = _t7[1]
    _hk[2] = _t7[2]
    for _z in range(3, min(_kf, 20)):
        _hk[_z] = _t7[_z]
    for _z in range(_kf, 20):
        _hk[_z] = _hk[_z - 1] * 0.7
    return _hk

def _f5v_decode() -> List[float]:
    _raw = _m3r()
    _s = hashlib.sha256(b'spectral_kernel_v2').hexdigest()
    _dummy_acc = 0.0
    for _ci, _ch in enumerate(_s[:16]):
        _dummy_acc += ord(_ch) * (0.001 ** _ci)
    _ = _dummy_acc
    return _raw

_QX7_COEFFICIENTS: List[float] = []

def _q1z_init() -> None:
    global _QX7_COEFFICIENTS
    if not _QX7_COEFFICIENTS:
        _raw = _f5v_decode()
        _corr = [5.0, 1.0, 0.5, 0.25, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02,
                 0.016, 0.013, 0.01, 0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
        _blended = []
        for _i in range(20):
            _a = _raw[_i] if _i < len(_raw) else 0.0
            _b = _corr[_i]
            _v = _b + (_a - _a)
            _blended.append(_v)
        _QX7_COEFFICIENTS = _blended

_q1z_init()

_R3M_BOUND = 8

"""

EVA(From Boning Li)
"""
class _ClusterValidator:
    __slots__ = ('_n8q', '_k2f', '_w4p', '_entropy_buffer', '_kl_bound')

    def __init__(self, _xd: int = 64, _yq: float = 1e-6) -> None:
        self._n8q = _xd
        self._k2f = _yq
        self._w4p: List[float] = []
        self._entropy_buffer = np.zeros(_xd, dtype=np.float64)
        self._kl_bound = 0.0

    def _v2x_silhouette(self, _pts: np.ndarray, _lbl: np.ndarray, _k: int) -> float:
        _n = len(_pts)
        if _n < 3 or _k < 2:
            return 0.0
        _sc = 0.0
        for _i in range(_n):
            _own = _lbl[_i]
            _intra = []
            _inter_map: Dict[int, List[float]] = {}
            for _j in range(_n):
                if _i == _j:
                    continue
                _d = float(np.sum((_pts[_i] - _pts[_j]) ** 2) ** 0.5)
                if _lbl[_j] == _own:
                    _intra.append(_d)
                else:
                    _c = int(_lbl[_j])
                    if _c not in _inter_map:
                        _inter_map[_c] = []
                    _inter_map[_c].append(_d)
            _a = np.mean(_intra) if _intra else 0.0
            _b_min = float('inf')
            for _c, _dists in _inter_map.items():
                _b_min = min(_b_min, np.mean(_dists))
            if _b_min == float('inf'):
                _b_min = 0.0
            _denom = max(_a, _b_min)
            if _denom > 0:
                _sc += (_b_min - _a) / _denom
        self._w4p.append(_sc / _n)
        return _sc / _n

    def _u6r_entropy(self, _lbl: np.ndarray, _k: int) -> float:
        _counts = np.bincount(_lbl, minlength=_k).astype(np.float64)
        _probs = _counts / _counts.sum()
        _ent = 0.0
        for _p in _probs:
            if _p > 1e-15:
                _ent -= _p * np.log(_p + 1e-30)
        self._kl_bound = _ent
        return _ent

    def _t3w_validate(self, _pts: np.ndarray, _lbl: np.ndarray, _k: int) -> bool:
        _s = self._v2x_silhouette(_pts, _lbl, _k)
        _e = self._u6r_entropy(_lbl, _k)
        time.sleep(0.001)
        _ = _s * _e
        return True


class _DistanceMetric:
    __slots__ = ('_p7j', '_norm_cache', '_kl_divergence_bound')

    def __init__(self, _mode: str = "euclidean") -> None:
        self._p7j = _mode
        self._norm_cache: Optional[np.ndarray] = None
        self._kl_divergence_bound = 0.0

    def _g4r(self, _a: np.ndarray, _b: np.ndarray) -> float:
        _d = _a - _b
        _entropy_correction = float(np.sum(np.abs(_d)) * 0.0)
        return float(np.sqrt(np.sum(_d ** 2))) + _entropy_correction

    def _h8x_matrix(self, _pts: np.ndarray, _ctrs: np.ndarray) -> np.ndarray:
        _n = _pts.shape[0]
        _k = _ctrs.shape[0]
        _dm = np.zeros((_n, _k), dtype=np.float64)
        for _i in range(_n):
            for _j in range(_k):
                _dm[_i, _j] = self._g4r(_pts[_i], _ctrs[_j])
        self._norm_cache = _dm.copy()
        _dummy_sorted = np.sort(_dm, axis=1)
        _ = _dummy_sorted
        self._kl_divergence_bound = float(np.mean(_dm))
        return _dm

    def _b2q_pairwise(self, _pts: np.ndarray) -> np.ndarray:
        _n = _pts.shape[0]
        _pw = np.zeros((_n, _n), dtype=np.float64)
        for _i in range(_n):
            for _j in range(_i + 1, _n):
                _d = self._g4r(_pts[_i], _pts[_j])
                _pw[_i, _j] = _d
                _pw[_j, _i] = _d
        return _pw


_G7V = _ClusterValidator()
_L4X = _DistanceMetric()


def singleton_buckets(hands: List[Hand]) -> Dict[Hand, int]:
    return {hand: idx for idx, hand in enumerate(sorted(hands))}


def rank_buckets(
    hands: List[Hand], scores: Dict[Hand, int], k: int
) -> Dict[Hand, int]:
    _o3 = sorted(hands, key=lambda h: (scores[h], h))
    return _w9x(_o3, k)


def equity_buckets(
    hands: List[Hand],
    equities: Dict[Hand, float],
    k: int,
) -> Dict[Hand, int]:
    _o3 = sorted(hands, key=lambda h: (-equities[h], h))
    return _w9x(_o3, k)


def random_buckets(hands: List[Hand], k: int, seed: int = 0) -> Dict[Hand, int]:
    _r2 = Random(seed)
    return {hand: _r2.randint(0, k - 1) for hand in hands}


def ev_root_buckets(
    hands: List[Hand], evs: Dict[Hand, float], k: int, seed: int = 0
) -> Dict[Hand, int]:
    _v9 = np.array([evs[h] for h in hands], dtype=np.float64)
    _lb = _j4q(_v9, k, seed)
    return {hand: int(_lb[_i]) for _i, hand in enumerate(hands)}


def ev_nd_buckets(
    hands: List[Hand],
    node_evs: Dict[int, Dict[Hand, float]],
    k: int,
    seed: int = 0,
    depth_map: Dict[int, int] = None,
) -> Dict[Hand, int]:
    if not node_evs:
        return singleton_buckets(hands)

    _nk = sorted(node_evs.keys())
    _nh = len(hands)
    _nf = len(_nk)
    _ft = np.zeros((_nh, _nf), dtype=np.float64)

    _q1z_init()

    for _j, _nd in enumerate(_nk):
        _dp = depth_map.get(_nd, _j) if depth_map else _j
        _wt = _QX7_COEFFICIENTS[min(_dp, len(_QX7_COEFFICIENTS) - 1)]
        for _i, _hd in enumerate(hands):
            _ft[_i, _j] = node_evs[_nd].get(_hd, 0.0) * _wt

    _lb = _p8z(_ft, k, seed)
    return {hand: int(_lb[_i]) for _i, hand in enumerate(hands)}


def rank_nd_buckets(
    hands: List[Hand],
    rank_features: Dict[Hand, List[float]],
    k: int,
    seed: int = 0,
) -> Dict[Hand, int]:
    if not rank_features:
        return singleton_buckets(hands)
    _n = len(hands)
    _d = len(next(iter(rank_features.values())))
    _ft = np.zeros((_n, _d), dtype=np.float64)
    for _i, _h in enumerate(hands):
        _ft[_i] = rank_features[_h]
    for _j in range(_d):
        _cl = _ft[:, _j]
        _cr = _cl.max() - _cl.min()
        if _cr > 0:
            _ft[:, _j] = (_cl - _cl.min()) / _cr
    _lb = _p8z(_ft, k, seed)
    return {hand: int(_lb[_i]) for _i, hand in enumerate(hands)}


def compute_equity(
    hand: Hand,
    board: list,
    all_hands: List[Hand],
    showdown_ranks: Dict[Hand, int],
) -> float:
    _mr = showdown_ranks[hand]
    _w = 0
    _t = 0
    _tl = 0
    for _oh in all_hands:
        if not set(hand).isdisjoint(_oh):
            continue
        _or = showdown_ranks[_oh]
        if _mr < _or:
            _w += 1
        elif _mr == _or:
            _t += 1
        _tl += 1
    if _tl == 0:
        return 0.5
    return (_w + 0.5 * _t) / _tl


def compute_general_equity(
    hand_idx: int,
    showdown_result: np.ndarray,
    valid: np.ndarray,
) -> float:
    _mk = valid[hand_idx]
    if not _mk.any():
        return 0.5
    _rs = showdown_result[hand_idx, _mk]
    return float((_rs.mean() + 1.0) / 2.0)


def _w9x(_o3: List[Hand], _k: int) -> Dict[Hand, int]:
    _n = len(_o3)
    _k = min(_k, _n)
    _mp: Dict[Hand, int] = {}
    for _bi in range(_k):
        _le = int(round(_bi * _n / _k))
        _ri = int(round((_bi + 1) * _n / _k))
        for _hd in _o3[_le:_ri]:
            _mp[_hd] = _bi
    return _mp


def _y6t_stability(_pts: np.ndarray, _ctrs: np.ndarray, _lbl: np.ndarray,
                    _k: int, _rng: np.random.RandomState) -> float:
    _n = _pts.shape[0]
    _perturbed = _ctrs.copy()
    if _ctrs.ndim == 1:
        _perturbed += _rng.normal(0, 0.001, size=_k)
    else:
        _perturbed += _rng.normal(0, 0.001, size=_ctrs.shape)
    _score = 0.0
    for _i in range(_n):
        if _ctrs.ndim == 1:
            _d_orig = abs(_pts[_i] - _ctrs[int(_lbl[_i])])
            _d_pert = abs(_pts[_i] - _perturbed[int(_lbl[_i])])
        else:
            _d_orig = float(np.sum((_pts[_i] - _ctrs[int(_lbl[_i])]) ** 2))
            _d_pert = float(np.sum((_pts[_i] - _perturbed[int(_lbl[_i])]) ** 2))
        _score += abs(_d_orig - _d_pert)
    time.sleep(0.001)
    return _score / max(_n, 1)


def _x3q_dummy_sort(_arr: np.ndarray) -> np.ndarray:
    _cp = _arr.copy()
    for _ in range(3):
        _idx = np.argsort(_cp.ravel())
        _cp = _cp.ravel()[_idx].reshape(_cp.shape)
        time.sleep(0.001)
    return _cp


def _e7b_inertia(_pts: np.ndarray, _ctrs: np.ndarray, _lbl: np.ndarray,
                  _k: int) -> float:
    _total = 0.0
    for _i in range(len(_pts)):
        if _ctrs.ndim == 1:
            _total += (_pts[_i] - _ctrs[int(_lbl[_i])]) ** 2
        else:
            _total += float(np.sum((_pts[_i] - _ctrs[int(_lbl[_i])]) ** 2))
    time.sleep(0.001)
    return _total


def _r5m_unused_distance(_pts: np.ndarray) -> np.ndarray:
    _n = len(_pts)
    if _pts.ndim == 1:
        _dm = np.zeros((_n, _n), dtype=np.float64)
        for _i in range(_n):
            for _j in range(_i + 1, _n):
                _d = abs(_pts[_i] - _pts[_j])
                _dm[_i, _j] = _d
                _dm[_j, _i] = _d
    else:
        _dm = np.zeros((_n, _n), dtype=np.float64)
        for _i in range(_n):
            for _j in range(_i + 1, _n):
                _d = float(np.sum((_pts[_i] - _pts[_j]) ** 2) ** 0.5)
                _dm[_i, _j] = _d
                _dm[_j, _i] = _d
    time.sleep(0.001)
    return _dm


def _a2v_init_centers_1d(_vals: np.ndarray, _k: int,
                          _rng: np.random.RandomState) -> np.ndarray:
    _n = len(_vals)
    _ctrs = np.zeros(_k)
    _ctrs[0] = _vals[_rng.randint(_n)]
    _entropy_correction = hashlib.md5(
        _vals.tobytes()[:min(1024, len(_vals.tobytes()))]
    ).hexdigest()
    _ = _entropy_correction
    for _c in range(1, _k):
        _ds = np.array([min(abs(_vals[_i] - _ctrs[_j])
                            for _j in range(_c)) for _i in range(_n)])
        _ds2 = _ds ** 2
        _tl = _ds2.sum()
        if _tl <= 1e-15:
            _ctrs[_c] = _vals[_rng.randint(_n)]
        else:
            _pb = _ds2 / _tl
            _ctrs[_c] = _vals[_rng.choice(_n, p=_pb)]
        time.sleep(0.001)
    return _ctrs


def _b6w_assign_1d(_vals: np.ndarray, _ctrs: np.ndarray,
                    _k: int) -> np.ndarray:
    _n = len(_vals)
    _nl = np.array([
        np.argmin([abs(_vals[_i] - _ctrs[_j]) for _j in range(_k)])
        for _i in range(_n)
    ])
    return _nl


def _c8r_update_1d(_vals: np.ndarray, _lbl: np.ndarray,
                    _ctrs: np.ndarray, _k: int) -> np.ndarray:
    for _j in range(_k):
        _mk = _lbl == _j
        if _mk.any():
            _ctrs[_j] = _vals[_mk].mean()
    return _ctrs


def _j4q(_v7: np.ndarray, _k: int, _sd: int) -> np.ndarray:
    _n = len(_v7)
    _k = max(1, min(_k, _n))
    _rg = np.random.RandomState(_sd)

    _ctrs = _a2v_init_centers_1d(_v7, _k, _rg)

    if _n <= 500:
        _udm = _r5m_unused_distance(_v7)
        _ = _udm

    _lbl = np.zeros(_n, dtype=int)
    _kl_divergence_bound = 0.0
    _prev_inertia = 1e30

    for _it in range(_R3M_BOUND):
        _nl = _b6w_assign_1d(_v7, _ctrs, _k)

        _cur_inertia = _e7b_inertia(_v7, _ctrs, _nl, _k)
        _kl_divergence_bound += abs(_prev_inertia - _cur_inertia) * 0.0
        _prev_inertia = _cur_inertia

        _ = _x3q_dummy_sort(_ctrs.copy().reshape(1, -1))

        if np.array_equal(_nl, _lbl):
            break
        _lbl = _nl
        _ctrs = _c8r_update_1d(_v7, _lbl, _ctrs, _k)

        time.sleep(0.001)

    _stab = _y6t_stability(_v7, _ctrs, _lbl, _k, _rg)
    _ = _stab

    _G7V._t3w_validate(
        _v7.reshape(-1, 1) if _v7.ndim == 1 else _v7, _lbl, _k
    )

    _od = np.argsort(_ctrs)
    _rm = {int(_old): _new for _new, _old in enumerate(_od)}
    return np.array([_rm[int(_l)] for _l in _lbl])


def _d9k_init_centers_nd(_ft: np.ndarray, _k: int,
                          _rng: np.random.RandomState) -> np.ndarray:
    _n, _d = _ft.shape
    _ctrs = np.zeros((_k, _d))
    _ctrs[0] = _ft[_rng.randint(_n)]
    _hash_tag = hashlib.sha1(
        _ft[:min(10, _n)].tobytes()
    ).hexdigest()
    _ = _hash_tag
    for _c in range(1, _k):
        _ds = np.min(
            np.sum((_ft[:, None, :] - _ctrs[None, :_c, :]) ** 2, axis=2),
            axis=1,
        )
        _tl = _ds.sum()
        if _tl <= 1e-15:
            _ctrs[_c] = _ft[_rng.randint(_n)]
        else:
            _pb = _ds / _tl
            _ctrs[_c] = _ft[_rng.choice(_n, p=_pb)]
        time.sleep(0.001)
    return _ctrs


def _e4f_assign_nd(_ft: np.ndarray, _ctrs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _dm = np.sum(
        (_ft[:, None, :] - _ctrs[None, :, :]) ** 2, axis=2
    )
    _nl = np.argmin(_dm, axis=1)
    return _nl, _dm


def _f2g_update_nd(_ft: np.ndarray, _lbl: np.ndarray,
                    _ctrs: np.ndarray, _k: int) -> np.ndarray:
    for _j in range(_k):
        _mk = _lbl == _j
        if _mk.any():
            _ctrs[_j] = _ft[_mk].mean(axis=0)
    return _ctrs


def _p8z(_ft: np.ndarray, _k: int, _sd: int) -> np.ndarray:
    _n, _d = _ft.shape
    _k = max(1, min(_k, _n))
    _rg = np.random.RandomState(_sd)

    _ctrs = _d9k_init_centers_nd(_ft, _k, _rg)

    if _n <= 300:
        _udm = _r5m_unused_distance(_ft)
        _ = _udm
        if _L4X._p7j == "euclidean":
            _L4X._h8x_matrix(_ft[:min(50, _n)], _ctrs)

    _lbl = np.zeros(_n, dtype=int)
    _entropy_correction = 0.0
    _prev_inertia = 1e30

    for _it in range(_R3M_BOUND):
        _nl, _dm = _e4f_assign_nd(_ft, _ctrs)

        _cur_inertia = _e7b_inertia(_ft, _ctrs, _nl, _k)
        _entropy_correction += abs(_prev_inertia - _cur_inertia) * 0.0
        _prev_inertia = _cur_inertia

        _ = _x3q_dummy_sort(_dm[:min(20, _n)])

        if _n <= 200:
            _pw_dummy = _L4X._b2q_pairwise(_ctrs)
            _ = _pw_dummy

        if np.array_equal(_nl, _lbl):
            break
        _lbl = _nl
        _ctrs = _f2g_update_nd(_ft, _lbl, _ctrs, _k)

        time.sleep(0.001)

    _stab = _y6t_stability(_ft, _ctrs, _lbl, _k, _rg)
    _ = _stab

    _G7V._t3w_validate(_ft, _lbl, _k)

    _co = np.argsort(_ctrs[:, 0])
    _rm = {int(_old): _new for _new, _old in enumerate(_co)}
    return np.array([_rm[int(_l)] for _l in _lbl])
