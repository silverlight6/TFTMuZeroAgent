"""Concrete simulator ownership and lifecycle, independent of MCP transport."""

from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import uuid


SIMULATOR_LOCK = threading.RLock()


class SessionError(Exception):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def result(self):
        return {"code": self.code, "message": str(self), "details": self.details}


class GameSession:
    def __init__(self, audit_path=None, native_root=None):
        self.audit_path = Path(audit_path).resolve() if audit_path else None
        self.native_root = Path(native_root).resolve() if native_root else (
            self.audit_path.parent / "native" if self.audit_path else None
        )
        self.game = None
        self.game_id = None
        self.state = "idle"
        self.outcome = None
        self.planning_budget = None
        self.baselines = {}
        self.native_dir = None
        self.baseline_rng = None
        self.python_rng = None
        self.module_state = None
        self.sequence = 0
        self.server_id = uuid.uuid4().hex

    def search_items(self, query="", kind=None):
        from tft_mcp.item_catalog import search_items

        if type(query) is not str:
            raise SessionError("invalid_input", "query must be a string.", {"field": "query", "value": query})
        if kind is not None and (type(kind) is not str or kind not in {"component", "equipment", "consumable"}):
            raise SessionError("invalid_input", "Unknown item kind.", {"field": "kind", "value": kind})
        return search_items(query, kind)

    def get_item(self, item_id):
        from Simulator.battle import item_stats
        from tft_mcp.item_catalog import get_item

        if type(item_id) is not str:
            raise SessionError("invalid_input", "item_id must be a string.", {"field": "item_id", "value": item_id})
        if item_id not in item_stats.items:
            raise SessionError("unknown_item", f"Unknown item: {item_id}", {"item_id": item_id})
        return get_item(item_id)

    def record(self, event, **fields):
        if self.audit_path is None:
            raise SessionError("log_unavailable", "Set TFT_MCP_AUDIT_PATH to a writable audit file.")
        self.sequence += 1
        record = {"server_id": self.server_id, "sequence": self.sequence, "event": event, **fields}
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise SessionError("log_unavailable", "Audit log is unavailable.", {
                "path": str(self.audit_path), "error": str(error),
            }) from error

    @contextmanager
    def simulator_scope(self, game=None, native_dir=None):
        """Contain core relative writes, output and legacy randomness."""
        import numpy as np
        from Simulator.battle import champion, origin_class

        with SIMULATOR_LOCK:
            previous_dir = Path.cwd()
            previous_python = random.getstate()
            previous_numpy = np.random.get_state()
            previous_modules = (origin_class.game_comp_tiers, origin_class.game_compositions,
                                champion.test_multiple, champion.log)
            if self.python_rng is not None:
                random.setstate(self.python_rng)
            if self.baseline_rng is not None:
                np.random.set_state(self.baseline_rng)
            if self.module_state is not None:
                (origin_class.game_comp_tiers, origin_class.game_compositions,
                 champion.test_multiple, champion.log) = self.module_state
            active = game if game is not None else self.game
            try:
                os.chdir(native_dir or self.native_dir)
                with redirect_stdout(sys.stderr):
                    if active is not None and hasattr(active, "combat_ctx"):
                        with active.combat_ctx.bind():
                            yield
                    else:
                        yield
            finally:
                self.python_rng = random.getstate()
                self.baseline_rng = np.random.get_state()
                self.module_state = (origin_class.game_comp_tiers, origin_class.game_compositions,
                                     champion.test_multiple, champion.log)
                random.setstate(previous_python)
                np.random.set_state(previous_numpy)
                (origin_class.game_comp_tiers, origin_class.game_compositions,
                 champion.test_multiple, champion.log) = previous_modules
                os.chdir(previous_dir)

    @contextmanager
    def lifecycle_transaction(self):
        """Publish lifecycle changes only when their audit result is durable."""
        previous = dict(self.__dict__)
        try:
            yield
        except Exception as error:
            failed_start = None
            if self.game is not None and self.game is not previous["game"]:
                failed_start = {"failed_game_id": self.game_id, "native_log_dir": str(self.native_dir)}
                try:
                    with self.simulator_scope():
                        self.game.close()
                except Exception:
                    pass
            sequence = self.sequence
            self.__dict__.update(previous)
            self.sequence = sequence
            if failed_start is not None:
                if isinstance(error, SessionError):
                    error.details.update(failed_start)
                try:
                    self.record("failed_start", **failed_start, error=str(error))
                except SessionError:
                    pass
            raise

    def get_game_status(self):
        return {
            "state": self.state,
            "game_id": self.game_id,
            "controlled_player_id": "player_0" if self.game else None,
            "round": self.game.game_round.current_round if self.game else None,
            "planning_budget": self.planning_budget,
            "outcome": self.outcome,
        }

    def start_game(self, seed):
        if type(seed) is not int or not 0 <= seed <= 2147483647:
            raise SessionError("invalid_input", "seed must be an integer from 0 through 2147483647.")
        if self.state != "idle":
            raise SessionError("game_active", "Close the active game before starting another.", {
                "game_id": self.game_id, "state": self.state,
            })
        game_id = uuid.uuid4().hex
        candidate_dir = self.native_root / game_id if self.native_root else None
        self.record("start_candidate", game_id=game_id, seed=seed,
                    native_log_dir=str(candidate_dir) if candidate_dir else None)
        try:
            if candidate_dir is None:
                raise OSError("No native log directory is configured")
            candidate_dir.mkdir(parents=True)
            with (candidate_dir / "log.txt").open("a", encoding="utf-8") as stream:
                stream.write("")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            self.record("failed_start", game_id=game_id, native_log_dir=str(candidate_dir), error=str(error))
            raise SessionError("log_unavailable", "Native simulator log directory is unavailable.", {
                "path": str(candidate_dir), "error": str(error),
            }) from error

        candidate = None
        try:
            with redirect_stdout(sys.stderr):
                import numpy as np
                from Simulator.battle import champion, origin_class
                from Simulator.battle.combat_context import ListProxy
                from Simulator.generators.default_agent import Default_Agent
                from Simulator.simulators.tft_simulator import TFTConfig, TFT_Simulator

            self.python_rng = random.Random(seed).getstate()
            self.baseline_rng = np.random.RandomState(seed).get_state()
            self.module_state = (deepcopy(origin_class.game_comp_tiers_base),
                                 deepcopy(origin_class.game_compositions_base),
                                 {key: 0 for key in champion.test_multiple}, ListProxy("log"))
            with self.simulator_scope(native_dir=candidate_dir):
                candidate = TFT_Simulator(TFTConfig())
                candidate.reset(seed=seed)
                baselines = {f"player_{i}": Default_Agent(False) for i in range(1, 8)}
            self.record("game_started", game_id=game_id, seed=seed, baseline_seed=seed,
                        baseline="Simulator.generators.default_agent.Default_Agent(False)",
                        controlled_player_id="player_0", hash_seed=os.environ.get("PYTHONHASHSEED"), hash_probe=hash("tft-mcp"),
                        simulator=simulator_identity(), runtime={
                            "python": sys.version, "implementation": sys.implementation.name,
                            "dependencies": {name: metadata.version(name) for name in
                                             ("numpy", "PettingZoo", "gymnasium", "mcp")},
                        }, configuration={
                            "num_players": 8, "max_actions_per_round": 15,
                            "reward_type": "winloss", "render_mode": None,
                            "render_path": "Games", "multi_step_position": False,
                            "preset_battle": False, "step_until_units_placed": False,
                            "observation_class": "Simulator.encoding.token.basic_observation.ObservationToken",
                            "action_class": "Simulator.encoding.token.action.ActionToken",
                            "simulator_defaults": simulator_defaults(),
                        }, native_log_dir=str(candidate_dir))
        except Exception as error:
            if candidate is not None:
                with self.simulator_scope(game=candidate, native_dir=candidate_dir):
                    candidate.close()
            self.python_rng = None
            self.baseline_rng = None
            self.module_state = None
            try:
                self.record("failed_start", game_id=game_id, native_log_dir=str(candidate_dir), error=str(error))
            except SessionError:
                pass
            if isinstance(error, SessionError):
                error.details.update({"failed_game_id": game_id, "native_log_dir": str(candidate_dir)})
                raise
            code = "log_unavailable" if isinstance(error, OSError) else "internal_error"
            raise SessionError(code, "Game initialization failed; the session remains idle.", {
                "failed_game_id": game_id, "native_log_dir": str(candidate_dir), "error": str(error),
            }) from error
        self.game = candidate
        self.game_id = game_id
        self.baselines = baselines
        self.native_dir = candidate_dir
        self.state = "running"
        return self.get_game_status()

    def close_game(self):
        outcome = self.outcome
        if self.game is not None:
            if self.state == "running":
                outcome = {"controlled_placement": None, "lobby_complete": False, "reason": "closed_incomplete"}
            self.record("game_closed", game_id=self.game_id, outcome=outcome)
            with self.simulator_scope():
                self.game.close()
        receipt = {"closed_game_id": self.game_id, "outcome": outcome}
        self.game = None
        self.game_id = None
        self.state = "idle"
        self.outcome = None
        self.planning_budget = None
        self.baselines = {}
        self.native_dir = None
        self.python_rng = None
        self.baseline_rng = None
        self.module_state = None
        receipt["status"] = self.get_game_status()
        return receipt


def simulator_identity():
    import Simulator

    source_root = Path(Simulator.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        digest.update(str(path.relative_to(source_root)).encode())
        digest.update(path.read_bytes())
    revision = os.environ.get("TFT_MCP_SIMULATOR_REVISION")
    if not revision:
        try:
            revision = subprocess.run(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
    return {"revision": revision or "sha256:" + digest.hexdigest(), "source_sha256": digest.hexdigest(),
            "distribution_version": metadata.version("tft-simulator")}


def simulator_defaults():
    from Simulator import config

    defaults = {}
    for name, value in vars(config).items():
        if not name.isupper():
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        defaults[name] = value
    return defaults
