"""Per-battle mutable state.  Replaces module-level combat globals.

Bind a context for the duration of env reset/step (and combat) so concurrent
in-process environments do not share ``que``, team lists, or ability counters.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from Simulator.rng import EnvRNG

_TRAIT_KEYS = (
    "cultist", "divine", "dusk", "elderwood", "enlightened", "exile", "ninja",
    "spirit", "the_boss", "warlord", "adept", "assassin", "brawler", "dazzler",
    "duelist", "emperor", "hunter", "keeper", "mage", "mystic", "shade",
    "sharpshooter", "vanguard", "fortune", "moonlight", "tormented",
)

_current: ContextVar[Optional["CombatContext"]] = ContextVar("combat_ctx", default=None)
_thread_fallback = threading.local()


def _empty_amounts() -> Dict[str, Dict[str, int]]:
    return {key: {"blue": 0, "red": 0} for key in _TRAIT_KEYS}


def _empty_field() -> List[List[Any]]:
    return [[None] * 7 for _ in range(8)]


@dataclass
class CombatContext:
    rng: EnvRNG = field(default_factory=EnvRNG.from_episode_seed)
    blue: list = field(default_factory=list)
    red: list = field(default_factory=list)
    que: list = field(default_factory=list)
    log: list = field(default_factory=list)
    milliseconds: int = 0
    damage_dealt: list = field(default_factory=list)
    damage_dealt_teams: dict = field(default_factory=lambda: {"blue": 0, "red": 0})
    galio_spawned: dict = field(default_factory=lambda: {"blue": False, "red": False})
    warlord_wins: dict = field(default_factory=lambda: {"blue": 0, "red": 0})
    field_coordinates: list = field(default_factory=_empty_field)

    kennen_hits: list = field(default_factory=list)
    lulu_targeted: list = field(default_factory=list)
    morgana_mr_list: list = field(default_factory=list)
    riven_counter: list = field(default_factory=list)
    riven_identifier_list: list = field(default_factory=list)
    vi_armor_list: list = field(default_factory=list)
    yone_list: list = field(default_factory=list)
    yone_checking: bool = False

    jhin_shots: list = field(default_factory=list)
    kalista_targets: list = field(default_factory=list)
    vayne_targets: list = field(default_factory=list)
    zed_counter: list = field(default_factory=list)

    bramble_vest_list: list = field(default_factory=list)
    deathblade_list: list = field(default_factory=list)
    frozen_heart_list: list = field(default_factory=list)
    gargoyle_stoneplate_list: list = field(default_factory=list)
    hextech_gunblade_list: list = field(default_factory=list)
    ionic_spark_list: list = field(default_factory=list)
    last_whisper_list: list = field(default_factory=list)
    statikk_shiv_list: list = field(default_factory=list)
    titans_resolve_list: list = field(default_factory=list)

    cultist_stars: dict = field(default_factory=lambda: {"blue": 0, "red": 0})
    total_health_teams: dict = field(default_factory=lambda: {"blue": 0, "red": 0})
    galio_spawn_time: dict = field(default_factory=lambda: {"blue": 0, "red": 0})
    amounts: dict = field(default_factory=_empty_amounts)
    divine_attack_list: list = field(default_factory=list)
    divine_list: list = field(default_factory=list)
    elderwood_list: dict = field(default_factory=lambda: {"blue": 0, "red": 0})
    spirit_list: list = field(default_factory=list)
    duelist_helper_list: list = field(default_factory=list)
    shade_helper_list: list = field(default_factory=list)

    def reset_combat(self) -> None:
        """Clear fight-local state.  Leaves the episode RNG stream intact."""
        rng = self.rng
        self.__dict__.update(CombatContext(rng=rng).__dict__)
        self.rng = rng

    def attach(self, unit) -> None:
        unit.ctx = self

    @contextmanager
    def bind(self) -> Iterator["CombatContext"]:
        token: Token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)


def get_ctx() -> CombatContext:
    bound = _current.get()
    if bound is not None:
        return bound
    ctx = getattr(_thread_fallback, "ctx", None)
    if ctx is None:
        ctx = CombatContext()
        _thread_fallback.ctx = ctx
    return ctx


def py_random():
    return get_ctx().rng.py


def np_random():
    return get_ctx().rng.np_api


class ListProxy:
    """Module-level name that always views a list on the current CombatContext."""

    def __init__(self, attr: str):
        object.__setattr__(self, "_attr", attr)

    def _data(self):
        return getattr(get_ctx(), self._attr)

    def append(self, item):
        self._data().append(item)

    def remove(self, item):
        self._data().remove(item)

    def pop(self, *args):
        return self._data().pop(*args)

    def sort(self, *args, **kwargs):
        self._data().sort(*args, **kwargs)

    def clear(self):
        self._data().clear()

    def copy(self):
        return list(self._data())

    def index(self, *args):
        return self._data().index(*args)

    def count(self, item):
        return self._data().count(item)

    def extend(self, items):
        self._data().extend(items)

    def insert(self, index, item):
        self._data().insert(index, item)

    def __iter__(self):
        return iter(self._data())

    def __len__(self):
        return len(self._data())

    def __getitem__(self, index):
        return self._data()[index]

    def __setitem__(self, index, value):
        self._data()[index] = value

    def __delitem__(self, index):
        del self._data()[index]

    def __contains__(self, item):
        return item in self._data()

    def __bool__(self):
        return bool(self._data())

    def __eq__(self, other):
        if isinstance(other, ListProxy):
            return self._data() == other._data()
        return self._data() == other

    def __add__(self, other):
        if isinstance(other, ListProxy):
            other = other._data()
        return self._data() + list(other)

    def __radd__(self, other):
        return list(other) + self._data()

    def __repr__(self):
        return repr(self._data())


class DictProxy:
    """Module-level name that always views a dict on the current CombatContext."""

    def __init__(self, attr: str):
        object.__setattr__(self, "_attr", attr)

    def _data(self):
        return getattr(get_ctx(), self._attr)

    def get(self, *args, **kwargs):
        return self._data().get(*args, **kwargs)

    def keys(self):
        return self._data().keys()

    def values(self):
        return self._data().values()

    def items(self):
        return self._data().items()

    def update(self, *args, **kwargs):
        self._data().update(*args, **kwargs)

    def clear(self):
        self._data().clear()

    def copy(self):
        return dict(self._data())

    def __getitem__(self, key):
        return self._data()[key]

    def __setitem__(self, key, value):
        self._data()[key] = value

    def __delitem__(self, key):
        del self._data()[key]

    def __iter__(self):
        return iter(self._data())

    def __len__(self):
        return len(self._data())

    def __contains__(self, key):
        return key in self._data()

    def __bool__(self):
        return bool(self._data())

    def __repr__(self):
        return repr(self._data())


class RandomProxy:
    """Drop-in for the ``random`` module, routed through the bound EnvRNG."""

    def randint(self, a, b):
        return get_ctx().rng.py.randint(a, b)

    def random(self):
        return get_ctx().rng.py.random()

    def choice(self, seq):
        return get_ctx().rng.py.choice(seq)

    def shuffle(self, x):
        get_ctx().rng.py.shuffle(x)

    def sample(self, population, k):
        return get_ctx().rng.py.sample(list(population), k)

    def seed(self, *_args, **_kwargs):
        return None


class NPRandomProxy:
    """Drop-in for ``numpy.random`` in simulator modules."""

    def randint(self, low, high=None, size=None):
        return get_ctx().rng.np_api.randint(low, high, size=size)

    def rand(self, *size):
        return get_ctx().rng.np_api.rand(*size)

    def choice(self, a, size=None, replace=True, p=None):
        return get_ctx().rng.np_api.choice(a, size=size, replace=replace, p=p)

    def seed(self, *_args, **_kwargs):
        return None


def install_episode(env, seed, options) -> dict:
    """Attach EnvRNG + CombatContext to an env from reset(seed, options)."""
    options = options or {}
    run_seed = options.get("run_seed")
    env.episode_seed = seed
    env.run_seed = run_seed
    env.rng = EnvRNG.from_episode_seed(seed)
    env.combat_ctx = CombatContext(rng=env.rng)
    return options


def bind_episode(env) -> CombatContext:
    ctx = getattr(env, "combat_ctx", None)
    if ctx is None:
        install_episode(env, None, {})
        ctx = env.combat_ctx
    return ctx


def merge_seed_info(info: Optional[dict], env) -> dict:
    payload = dict(info or {})
    payload["episode_seed"] = getattr(env, "episode_seed", None)
    payload["run_seed"] = getattr(env, "run_seed", None)
    return payload
