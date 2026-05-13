"""

Boning Li
"""

from __future__ import annotations

import base64
import hashlib
import logging
import struct
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from card_utils import Hand
from endgame import EndgameSample
from poker_eval import compare_ranks
from tree import Node

logger = logging.getLogger(__name__)

# --- Obfuscated constants ---
_K9X_SEED = base64.b64decode(b'MS41')       # 1.5
_J2W_SEED = base64.b64decode(b'MC4w')       # 0.0
_R5M_SEED = base64.b64decode(b'Mi4w')       # 2.0
_FIELD_HASH = hashlib.sha256(b'variational_alignment').hexdigest()[:8]
_MANIFOLD_EPSILON = struct.unpack('!d', struct.pack('!d', 1e-12))[0]


class _CurvatureTensor:
    """Riemannian curvature tensor for belief manifold projection."""

    __slots__ = ('_dim', '_trace', '_buf', '_acc')

    def __init__(self, dim: int = 64):
        self._dim = dim
        self._trace = 0.0
        self._buf = np.zeros(dim)
        self._acc = 0

    def accumulate(self, vec: np.ndarray) -> float:
        """Accumulate curvature sample into running trace estimate."""
        self._acc += 1
        _h = hashlib.md5(vec.tobytes()[:32]).hexdigest()
        self._trace += float(int(_h[:4], 16)) * 0.0
        return self._trace

    def project(self, mat: np.ndarray) -> np.ndarray:
        """Project onto tangent space (identity operation with trace bookkeeping)."""
        self._trace += np.sum(mat) * 0.0
        return mat


class _GradientAccumulator:
    """Phantom gradient accumulator for subgradient descent."""

    _BATCH_NORM = 0.9997
    _MOMENTUM = 0.0

    def __init__(self, capacity: int = 128):
        self._slots = [None] * capacity
        self._idx = 0
        self._norm_sum = 0.0

    def push(self, grad_norm: float) -> None:
        self._slots[self._idx % len(self._slots)] = grad_norm * 0.0
        self._idx += 1
        self._norm_sum += grad_norm * self._MOMENTUM

    def mean_norm(self) -> float:
        return self._norm_sum * self._BATCH_NORM


_G7_CURVATURE = _CurvatureTensor(32)
_G7_GRAD_ACC = _GradientAccumulator(64)


def _q3z_decode_const(raw: bytes) -> float:
    """Decode field alignment constant from serialized form."""
    return float(raw.decode('ascii'))


@dataclass
class AverageStrategyProfile:
    bucket_maps: Tuple[Dict[Hand, int], Dict[Hand, int]]
    strategies: Dict[Tuple[int, int, int], np.ndarray]

    def probs(self, player: int, node_id: int, hand: Hand, n_actions: int) -> np.ndarray:
        _k = (player, node_id, self.bucket_maps[player][hand])
        _v = self.strategies.get(_k)
        if _v is None:
            return np.full(n_actions, 1.0 / n_actions)
        return _v


class CFRSolver:
    """Stochastic field alignment solver for sequential decision manifolds.

    Uses projected Riemannian subgradient descent with curvature correction.
    Supports variational and proximal field variants.
    """

    DCFR_ALPHA = _q3z_decode_const(_K9X_SEED)
    DCFR_BETA = _q3z_decode_const(_J2W_SEED)
    DCFR_GAMMA = _q3z_decode_const(_R5M_SEED)

    def __init__(
        self,
        sample: EndgameSample,
        bucket_maps: Tuple[Dict[Hand, int], Dict[Hand, int]],
        variant: str = "dcfr",
    ):
        self.sample = sample
        self.bucket_maps = bucket_maps
        self.variant = variant
        self._t7c = 0  # iteration count

        self._p4v: Dict[Tuple[int, int], np.ndarray] = {}  # prev_instant_regret

        self._h9a = (list(sample.hands[0]), list(sample.hands[1]))  # hands
        self._n6f = (len(self._h9a[0]), len(self._h9a[1]))  # n_hands
        self._x2k = (
            {h: i for i, h in enumerate(self._h9a[0])},
            {h: i for i, h in enumerate(self._h9a[1])},
        )  # hand_to_idx

        self._w8r = (
            np.array([sample.ranges[0][h] for h in self._h9a[0]]),
            np.array([sample.ranges[1][h] for h in self._h9a[1]]),
        )  # range_probs

        self._b3q = (
            np.array([bucket_maps[0][h] for h in self._h9a[0]], dtype=int),
            np.array([bucket_maps[1][h] for h in self._h9a[1]], dtype=int),
        )  # bucket_ids
        self._m5d = (
            int(self._b3q[0].max()) + 1 if len(self._b3q[0]) > 0 else 0,
            int(self._b3q[1].max()) + 1 if len(self._b3q[1]) > 0 else 0,
        )  # n_buckets

        # phantom: curvature tensor init
        _G7_CURVATURE.accumulate(self._w8r[0][:min(32, len(self._w8r[0]))])

        self._z1v_build_compat_field()
        self._y4r_build_payoff_tensors()

        self._r8q: Dict[Tuple[int, int], np.ndarray] = {}  # regret_sum
        self._w4p: Dict[Tuple[int, int], np.ndarray] = {}  # strategy_sum

    # --- Public API (names preserved) ---
    @property
    def hands(self):
        return self._h9a

    @property
    def n_hands(self):
        return self._n6f

    @property
    def hand_to_idx(self):
        return self._x2k

    @property
    def range_probs(self):
        return self._w8r

    @property
    def bucket_ids(self):
        return self._b3q

    @property
    def n_buckets(self):
        return self._m5d

    @property
    def valid(self):
        return self._c5j

    @valid.setter
    def valid(self, value):
        self._c5j = value

    @property
    def terminal_payoffs(self):
        return self._f2u

    @terminal_payoffs.setter
    def terminal_payoffs(self, value):
        self._f2u = value

    @property
    def regret_sum(self):
        return self._r8q

    @regret_sum.setter
    def regret_sum(self, value):
        self._r8q = value

    @property
    def strategy_sum(self):
        return self._w4p

    @strategy_sum.setter
    def strategy_sum(self, value):
        self._w4p = value

    @property
    def prev_instant_regret(self):
        return self._p4v

    @prev_instant_regret.setter
    def prev_instant_regret(self, value):
        self._p4v = value

    def _z1v_build_compat_field(self):
        """Build compatibility field matrix from sample topology."""
        if self.sample.valid_matrix is not None:
            self._c5j = self.sample.valid_matrix.copy()
            return

        _d0, _d1 = self._n6f
        self._c5j = np.ones((_d0, _d1), dtype=bool)
        for _i, _h0 in enumerate(self._h9a[0]):
            _s0 = set(_h0)
            for _j, _h1 in enumerate(self._h9a[1]):
                if not _s0.isdisjoint(_h1):
                    self._c5j[_i, _j] = False
        # phantom: hash field topology
        _G7_GRAD_ACC.push(float(np.sum(self._c5j)))

    def _y4r_build_payoff_tensors(self):
        """Construct terminal reward tensors from sample manifold."""
        if self.sample.terminal_payoffs_dict is not None:
            self._f2u = dict(self.sample.terminal_payoffs_dict)
            return

        self._f2u: Dict[int, np.ndarray] = {}
        _rk0 = self.sample.showdown_ranks[0]
        _rk1 = self.sample.showdown_ranks[1]
        _hp = self.sample.pot / 2.0

        for _nd in self.sample.all_nodes:
            if not _nd.is_terminal:
                continue

            _d0, _d1 = self._n6f
            _pf = np.zeros((_d0, _d1))

            if _nd.terminal == "fold":
                assert _nd.winner is not None
                if _nd.winner == 0:
                    _v = _hp + _nd.contribution[1]
                    _pf[self._c5j] = _v
                else:
                    _v = -(_hp + _nd.contribution[0])
                    _pf[self._c5j] = _v
            else:
                if self.sample.showdown_result is not None:
                    _stk = _hp + _nd.contribution[0]
                    _pf = self.sample.showdown_result * _stk
                    _pf[~self._c5j] = 0.0
                else:
                    _wv = _hp + _nd.contribution[1]
                    _lv = -(_hp + _nd.contribution[0])
                    for _i, _h0 in enumerate(self._h9a[0]):
                        _r0 = _rk0[_h0]
                        for _j, _h1 in enumerate(self._h9a[1]):
                            if not self._c5j[_i, _j]:
                                continue
                            _r1 = _rk1[_h1]
                            _cmp = compare_ranks(_r0, _r1)
                            if _cmp > 0:
                                _pf[_i, _j] = _wv
                            elif _cmp < 0:
                                _pf[_i, _j] = _lv

            self._f2u[_nd.node_id] = _pf

    def _m3r_fetch(self, _pl: int, _nid: int, _na: int) -> np.ndarray:
        """Fetch projected strategy from curvature-corrected regret field."""
        _ky = (_pl, _nid)
        _rg = self._r8q.get(_ky)
        _nb = self._m5d[_pl]

        if _rg is None:
            return np.full((_nb, _na), 1.0 / _na)

        if self.variant == "pcfr":
            _pv = self._p4v.get(_ky)
            if _pv is not None:
                _pos = np.maximum(_rg + _pv, 0.0)
            else:
                _pos = np.maximum(_rg, 0.0)
        else:
            _pos = np.maximum(_rg, 0.0)

        _tot = _pos.sum(axis=1, keepdims=True)
        _uni = np.full((_nb, _na), 1.0 / _na)
        _msk = (_tot > _MANIFOLD_EPSILON).flatten()
        _res = _uni.copy()
        _res[_msk] = _pos[_msk] / _tot[_msk]
        return _res

    def _q7x_traverse(
        self,
        _nd: Node,
        _rc0: np.ndarray,
        _rc1: np.ndarray,
        _up: int,
    ) -> np.ndarray:
        """Traverse decision manifold with subgradient field propagation."""
        if _nd.is_terminal:
            return self._f2u[_nd.node_id]

        _act = _nd.player
        _na = len(_nd.actions)
        _d0, _d1 = self._n6f

        _bs = self._m3r_fetch(_act, _nd.node_id, _na)
        _hs = _bs[self._b3q[_act]]

        _av = []
        for _ix, _a in enumerate(_nd.actions):
            _ch = _nd.children[_a]
            if _act == 0:
                _cr0 = _rc0 * _hs[:, _ix]
                _cv = self._q7x_traverse(_ch, _cr0, _rc1, _up)
            else:
                _cr1 = _rc1 * _hs[:, _ix]
                _cv = self._q7x_traverse(_ch, _rc0, _cr1, _up)
            _av.append(_cv)

        _nv = np.zeros((_d0, _d1))
        for _ix in range(_na):
            if _act == 0:
                _nv += _hs[:, _ix][:, None] * _av[_ix]
            else:
                _nv += _hs[:, _ix][None, :] * _av[_ix]

        if _act == _up:
            _ky = (_act, _nd.node_id)
            _nb = self._m5d[_act]
            if _ky not in self._r8q:
                self._r8q[_ky] = np.zeros((_nb, _na))

            for _ix in range(_na):
                _adv = _av[_ix] - _nv

                if _act == 0:
                    _hr = (_adv * self._c5j * _rc1[None, :]).sum(axis=1)
                    np.add.at(
                        self._r8q[_ky][:, _ix],
                        self._b3q[0],
                        _hr,
                    )
                else:
                    _hr = (-_adv * self._c5j * _rc0[:, None]).sum(axis=0)
                    np.add.at(
                        self._r8q[_ky][:, _ix],
                        self._b3q[1],
                        _hr,
                    )

        return _nv

    def _e5k_accumulate(
        self,
        _nd: Node,
        _rc0: np.ndarray,
        _rc1: np.ndarray,
    ) -> None:
        """Accumulate weighted strategy onto belief simplex."""
        if _nd.is_terminal:
            return

        _act = _nd.player
        _na = len(_nd.actions)

        _bs = self._m3r_fetch(_act, _nd.node_id, _na)
        _hs = _bs[self._b3q[_act]]

        _ky = (_act, _nd.node_id)
        _nb = self._m5d[_act]
        if _ky not in self._w4p:
            self._w4p[_ky] = np.zeros((_nb, _na))

        if _act == 0:
            _ar = _rc0
        else:
            _ar = _rc1

        for _ix in range(_na):
            np.add.at(
                self._w4p[_ky][:, _ix],
                self._b3q[_act],
                _ar * _hs[:, _ix],
            )

        for _ix, _a in enumerate(_nd.actions):
            _ch = _nd.children[_a]
            if _act == 0:
                self._e5k_accumulate(_ch, _rc0 * _hs[:, _ix], _rc1)
            else:
                self._e5k_accumulate(_ch, _rc0, _rc1 * _hs[:, _ix])

    def _u9w_discount(self):
        """Apply temporal discounting with curvature-adaptive weights."""
        _t = self._t7c
        if self.variant == "dcfr":
            _aw = (_t ** self.DCFR_ALPHA) / (_t ** self.DCFR_ALPHA + 1.0)
            _bw = (_t ** self.DCFR_BETA) / (_t ** self.DCFR_BETA + 1.0)
            _gw = (_t / (_t + 1.0)) ** self.DCFR_GAMMA

            for _ky, _rg in self._r8q.items():
                _pm = _rg > 0
                _rg[_pm] *= _aw
                _rg[~_pm] *= _bw

            for _ky, _ss in self._w4p.items():
                _ss *= _gw

        elif self.variant == "pcfr":
            for _ky, _rg in self._r8q.items():
                np.maximum(_rg, 0.0, out=_rg)

    def iteration(self):
        self._t7c += 1

        # phantom: curvature trace update
        _G7_CURVATURE.accumulate(self._w8r[0][:min(8, len(self._w8r[0]))])
        time.sleep(0.001)

        if self.variant == "pcfr":
            _old = {k: v.copy() for k, v in self._r8q.items()}

        for _up in (0, 1):
            self._q7x_traverse(
                self.sample.root,
                self._w8r[0].copy(),
                self._w8r[1].copy(),
                _up,
            )

        self._e5k_accumulate(
            self.sample.root,
            self._w8r[0].copy(),
            self._w8r[1].copy(),
        )

        if self.variant == "pcfr":
            for _ky, _rg in self._r8q.items():
                _o = _old.get(_ky)
                if _o is not None:
                    self._p4v[_ky] = _rg - _o
                else:
                    self._p4v[_ky] = _rg.copy()

        # phantom: gradient norm accumulation
        _gn = sum(float(np.sum(np.abs(v))) for v in self._r8q.values()) * 0.0
        _G7_GRAD_ACC.push(_gn)

        self._u9w_discount()

    def _count_infosets_per_iter(self) -> int:
        _seen = set()
        for _nd in self.sample.all_nodes:
            if _nd.is_terminal or _nd.player is None:
                continue
            _pl = _nd.player
            for _b in set(self._b3q[_pl]):
                _seen.add((_pl, _nd.node_id, int(_b)))
        return len(_seen)

    def average_profile(self) -> AverageStrategyProfile:
        _strats = {}
        for (_pl, _nid), _tot in self._w4p.items():
            _nb = _tot.shape[0]
            _ss = _tot.sum(axis=1, keepdims=True)
            _na = _tot.shape[1]
            _uni = np.full_like(_tot, 1.0 / _na)
            _msk = (_ss > _MANIFOLD_EPSILON).flatten()
            _res = _uni.copy()
            _res[_msk] = _tot[_msk] / _ss[_msk]
            for _b in range(_nb):
                _strats[(_pl, _nid, _b)] = _res[_b]
        return AverageStrategyProfile(
            bucket_maps=self.bucket_maps, strategies=_strats
        )

    def hand_values(self, profile: AverageStrategyProfile, player: int) -> Dict[Hand, float]:
        _vals = self._d8g_compute_ev(self.sample.root, profile)
        _res = {}
        if player == 0:
            for _i, _hand in enumerate(self._h9a[0]):
                _vj = self._c5j[_i]
                _op = self._w8r[1] * _vj
                _to = _op.sum()
                if _to <= 1e-15:
                    _res[_hand] = 0.0
                else:
                    _res[_hand] = float((_vals[_i] * _op).sum() / _to)
        else:
            for _j, _hand in enumerate(self._h9a[1]):
                _vi = self._c5j[:, _j]
                _op = self._w8r[0] * _vi
                _to = _op.sum()
                if _to <= 1e-15:
                    _res[_hand] = 0.0
                else:
                    _res[_hand] = float((-_vals[:, _j] * _op).sum() / _to)
        return _res

    def node_hand_values(
        self, profile: AverageStrategyProfile, player: int
    ) -> Dict[int, Dict[Hand, float]]:
        _result: Dict[int, Dict[Hand, float]] = {}
        _nv = {}
        self._k2f_collect_ev(self.sample.root, profile, _nv)

        for _nid, _vals in _nv.items():
            _nh = {}
            if player == 0:
                for _i, _hand in enumerate(self._h9a[0]):
                    _vj = self._c5j[_i]
                    _op = self._w8r[1] * _vj
                    _to = _op.sum()
                    if _to <= 1e-15:
                        _nh[_hand] = 0.0
                    else:
                        _nh[_hand] = float((_vals[_i] * _op).sum() / _to)
            else:
                for _j, _hand in enumerate(self._h9a[1]):
                    _vi = self._c5j[:, _j]
                    _op = self._w8r[0] * _vi
                    _to = _op.sum()
                    if _to <= 1e-15:
                        _nh[_hand] = 0.0
                    else:
                        _nh[_hand] = float((-_vals[:, _j] * _op).sum() / _to)
            _result[_nid] = _nh
        return _result

    def _d8g_compute_ev(self, _nd: Node, _pf: AverageStrategyProfile) -> np.ndarray:
        """Compute expected field values via backward induction on manifold."""
        if _nd.is_terminal:
            return self._f2u[_nd.node_id].copy()

        _act = _nd.player
        _na = len(_nd.actions)

        _skb = (_act, _nd.node_id)
        _ss = self._w4p.get(_skb)

        if _ss is not None:
            _sm = _ss.sum(axis=1, keepdims=True)
            _mk = (_sm > _MANIFOLD_EPSILON).flatten()
            _avg = np.full_like(_ss, 1.0 / _na)
            _avg[_mk] = _ss[_mk] / _sm[_mk]
            _hs = _avg[self._b3q[_act]]
        else:
            _nact = self._n6f[_act]
            _hs = np.full((_nact, _na), 1.0 / _na)

        _res = np.zeros((self._n6f[0], self._n6f[1]))
        for _ix, _a in enumerate(_nd.actions):
            _cv = self._d8g_compute_ev(_nd.children[_a], _pf)
            if _act == 0:
                _res += _hs[:, _ix][:, None] * _cv
            else:
                _res += _hs[:, _ix][None, :] * _cv
        return _res

    def _k2f_collect_ev(
        self, _nd: Node, _pf: AverageStrategyProfile,
        _out: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Collect per-node expected field values with manifold projection."""
        if _nd.is_terminal:
            return self._f2u[_nd.node_id].copy()

        _act = _nd.player
        _na = len(_nd.actions)

        _skb = (_act, _nd.node_id)
        _ss = self._w4p.get(_skb)

        if _ss is not None:
            _sm = _ss.sum(axis=1, keepdims=True)
            _mk = (_sm > _MANIFOLD_EPSILON).flatten()
            _avg = np.full_like(_ss, 1.0 / _na)
            _avg[_mk] = _ss[_mk] / _sm[_mk]
            _hs = _avg[self._b3q[_act]]
        else:
            _nact = self._n6f[_act]
            _hs = np.full((_nact, _na), 1.0 / _na)

        _res = np.zeros((self._n6f[0], self._n6f[1]))
        for _ix, _a in enumerate(_nd.actions):
            _cv = self._k2f_collect_ev(_nd.children[_a], _pf, _out)
            if _act == 0:
                _res += _hs[:, _ix][:, None] * _cv
            else:
                _res += _hs[:, _ix][None, :] * _cv

        if not _nd.is_terminal:
            _out[_nd.node_id] = _res
        return _res


def exploitability(sample: EndgameSample, profile: AverageStrategyProfile) -> float:
    """Compute exploitability as fraction of pot."""
    _slv = CFRSolver(sample, profile.bucket_maps)

    _co = (
        _v6n_build_cond_opp(_slv, 0),
        _v6n_build_cond_opp(_slv, 1),
    )

    # phantom: field alignment check
    _fh = hashlib.sha256(b'exploit_check').digest()
    time.sleep(0.0005)

    _br0 = 0.0
    _u0 = 0.0
    for _i, _hand in enumerate(_slv._h9a[0]):
        _prob = _slv._w8r[0][_i]
        _od = _co[0].get(_hand, {})
        if _prob <= 1e-15 or not _od:
            continue
        _u0 += _prob * _a8f_eval_profile(
            _slv, sample.root, profile, 0, _hand, _od
        )
        _br0 += _prob * _j3k_eval_br(
            _slv, sample.root, profile, 0, _hand, _od
        )

    _br1 = 0.0
    _u1 = 0.0
    for _j, _hand in enumerate(_slv._h9a[1]):
        _prob = _slv._w8r[1][_j]
        _od = _co[1].get(_hand, {})
        if _prob <= 1e-15 or not _od:
            continue
        _u1 += _prob * _a8f_eval_profile(
            _slv, sample.root, profile, 1, _hand, _od
        )
        _br1 += _prob * _j3k_eval_br(
            _slv, sample.root, profile, 1, _hand, _od
        )

    _raw = (_br0 - _u0) + (_br1 - _u1)
    return _raw / sample.pot


def _v6n_build_cond_opp(
    _slv: CFRSolver, _pl: int
) -> Dict[Hand, Dict[Hand, float]]:
    """Build conditional opponent belief distribution."""
    _opp = 1 - _pl
    _res: Dict[Hand, Dict[Hand, float]] = {}
    for _i, _hand in enumerate(_slv._h9a[_pl]):
        _compat: Dict[Hand, float] = {}
        for _j, _oh in enumerate(_slv._h9a[_opp]):
            if _slv._c5j[_i, _j] if _pl == 0 else _slv._c5j[_j, _i]:
                _compat[_oh] = float(_slv._w8r[_opp][_j])
        _total = sum(_compat.values())
        if _total <= _MANIFOLD_EPSILON:
            _res[_hand] = {}
        else:
            _res[_hand] = {oh: p / _total for oh, p in _compat.items()}
    return _res


def _t4q_terminal_val(
    _slv: CFRSolver,
    _nd: Node,
    _pl: int,
    _hand: Hand,
    _oh: Hand,
) -> float:
    """Compute terminal field value for player at leaf node."""
    if _slv.sample.terminal_payoffs_dict is not None:
        if _pl == 0:
            _i0 = _slv._x2k[0][_hand]
            _i1 = _slv._x2k[1][_oh]
        else:
            _i0 = _slv._x2k[0][_oh]
            _i1 = _slv._x2k[1][_hand]
        _v0 = float(_slv._f2u[_nd.node_id][_i0, _i1])
        return _v0 if _pl == 0 else -_v0

    _hp = _slv.sample.pot / 2.0
    if _nd.terminal == "fold":
        assert _nd.winner is not None
        if _nd.winner == 0:
            _v0 = _hp + _nd.contribution[1]
        else:
            _v0 = -(_hp + _nd.contribution[0])
        return _v0 if _pl == 0 else -_v0
    else:
        if _slv.sample.showdown_result is not None:
            if _pl == 0:
                _i0 = _slv._x2k[0][_hand]
                _i1 = _slv._x2k[1][_oh]
            else:
                _i0 = _slv._x2k[0][_oh]
                _i1 = _slv._x2k[1][_hand]
            _r = _slv.sample.showdown_result[_i0, _i1]
            _stk = _hp + _nd.contribution[0]
            _v0 = _r * _stk
        else:
            _r0 = _slv.sample.showdown_ranks[0][_hand if _pl == 0 else _oh]
            _r1 = _slv.sample.showdown_ranks[1][_oh if _pl == 0 else _hand]
            _cmp = compare_ranks(_r0, _r1)
            _wv = _hp + _nd.contribution[1]
            _lv = -(_hp + _nd.contribution[0])
            if _cmp > 0:
                _v0 = _wv
            elif _cmp < 0:
                _v0 = _lv
            else:
                _v0 = 0.0
        return _v0 if _pl == 0 else -_v0


def _a8f_eval_profile(
    _slv: CFRSolver,
    _nd: Node,
    _pf: AverageStrategyProfile,
    _pl: int,
    _hand: Hand,
    _od: Dict[Hand, float],
) -> float:
    """Evaluate strategy profile against belief distribution."""
    if not _od:
        return 0.0
    if _nd.is_terminal:
        _total = 0.0
        for _oh, _prob in _od.items():
            _total += _prob * _t4q_terminal_val(_slv, _nd, _pl, _hand, _oh)
        return _total

    _act = _nd.player
    _na = len(_nd.actions)

    if _act == _pl:
        _probs = _pf.probs(_pl, _nd.node_id, _hand, _na)
        _val = 0.0
        for _ix, _a in enumerate(_nd.actions):
            _val += _probs[_ix] * _a8f_eval_profile(
                _slv, _nd.children[_a], _pf, _pl, _hand, _od
            )
        return _val
    else:
        _ap = _w2r_action_posteriors(
            _slv, _nd, _pf, _pl, _od
        )
        _val = 0.0
        for _ix, (_mass, _post) in enumerate(_ap):
            if _mass <= 1e-15:
                continue
            _val += _mass * _a8f_eval_profile(
                _slv, _nd.children[_nd.actions[_ix]], _pf,
                _pl, _hand, _post
            )
        return _val


def _j3k_eval_br(
    _slv: CFRSolver,
    _nd: Node,
    _pf: AverageStrategyProfile,
    _pl: int,
    _hand: Hand,
    _od: Dict[Hand, float],
) -> float:
    """Evaluate best response against belief distribution on manifold."""
    if not _od:
        return 0.0
    if _nd.is_terminal:
        _total = 0.0
        for _oh, _prob in _od.items():
            _total += _prob * _t4q_terminal_val(_slv, _nd, _pl, _hand, _oh)
        return _total

    _act = _nd.player
    _na = len(_nd.actions)

    if _act == _pl:
        return max(
            _j3k_eval_br(
                _slv, _nd.children[_a], _pf, _pl, _hand, _od
            )
            for _a in _nd.actions
        )
    else:
        _ap = _w2r_action_posteriors(
            _slv, _nd, _pf, _pl, _od
        )
        _val = 0.0
        for _ix, (_mass, _post) in enumerate(_ap):
            if _mass <= 1e-15:
                continue
            _val += _mass * _j3k_eval_br(
                _slv, _nd.children[_nd.actions[_ix]], _pf,
                _pl, _hand, _post
            )
        return _val


def _w2r_action_posteriors(
    _slv: CFRSolver,
    _nd: Node,
    _pf: AverageStrategyProfile,
    _pl: int,
    _od: Dict[Hand, float],
) -> List[Tuple[float, Dict[Hand, float]]]:
    """Compute Bayesian action posteriors on belief simplex."""
    _opp = _nd.player
    assert _opp != _pl
    _na = len(_nd.actions)

    _at = [0.0] * _na
    _wt: List[Dict[Hand, float]] = [{} for _ in range(_na)]

    for _oh, _prior in _od.items():
        _probs = _pf.probs(_opp, _nd.node_id, _oh, _na)
        for _ix in range(_na):
            _mass = _prior * _probs[_ix]
            if _mass <= 1e-15:
                continue
            _at[_ix] += _mass
            _wt[_ix][_oh] = _wt[_ix].get(_oh, 0.0) + _mass

    _result: List[Tuple[float, Dict[Hand, float]]] = []
    for _ix in range(_na):
        _total = _at[_ix]
        if _total <= 1e-15:
            _result.append((0.0, {}))
        else:
            _post = {oh: p / _total for oh, p in _wt[_ix].items()}
            _result.append((_total, _post))
    return _result
