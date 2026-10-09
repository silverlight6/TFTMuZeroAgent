"""Concrete simulator ownership and lifecycle, independent of MCP transport."""

from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from functools import wraps
import subprocess
import sys
import threading
import uuid


SIMULATOR_LOCK = threading.RLock()
MAX_PROGRESS_STEPS = 6000


class SessionError(Exception):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def result(self):
        return {"code": self.code, "message": str(self), "details": self.details}


def transactional(method):
    @wraps(method)
    def invoke(self, *args, **kwargs):
        with self.lifecycle_transaction(mutable=True):
            return method(self, *args, **kwargs)
    return invoke


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
        self.placements = {}
        self.terminal_snapshot = None
        self._records = None
        self._candidate_prepared = False
        self.public_final = {}
        self.sequence = 0
        self.server_id = uuid.uuid4().hex

    def search_champions(self, query='', cost=None, trait_id=None):
        from tft_mcp import champion_catalog
        if trait_id is not None and trait_id not in champion_catalog.origin_class_stats.tiers:
            raise SessionError('invalid_input', 'Unknown trait filter.', {'field': 'trait_id', 'value': trait_id})
        return champion_catalog.search_champions(query, cost, trait_id)

    def get_champion(self, champion_id):
        from tft_mcp import champion_catalog
        if champion_id not in champion_catalog.stats.BASE_CHAMPION_LIST:
            raise SessionError('unknown_champion', f'Unknown champion: {champion_id}.', {'champion_id': champion_id})
        return champion_catalog.get_champion(champion_id)

    def search_traits(self, query=''):
        from tft_mcp import champion_catalog
        return champion_catalog.search_traits(query)

    def get_trait(self, trait_id):
        from tft_mcp import champion_catalog
        if trait_id not in champion_catalog.origin_class_stats.tiers:
            raise SessionError('unknown_trait', f'Unknown trait: {trait_id}.', {'trait_id': trait_id})
        return champion_catalog.get_trait(trait_id)

    def get_trait_champions(self, trait_id):
        from tft_mcp import champion_catalog
        if trait_id not in champion_catalog.origin_class_stats.tiers:
            raise SessionError('unknown_trait', f'Unknown trait: {trait_id}.', {'trait_id': trait_id})
        return champion_catalog.get_trait_champions(trait_id)

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
        if self._records is None:
            with self.lifecycle_transaction():
                self.record(event, **fields)
            return
        self.sequence += 1
        self._records.append({"server_id": self.server_id, "sequence": self.sequence,
                              "event": event, **fields})

    def probe_audit(self):
        if self.audit_path is None:
            raise SessionError("log_unavailable", "Set TFT_MCP_AUDIT_PATH to a writable audit file.")
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            if self.audit_path.exists():
                self.audit_path.read_bytes()
            with tempfile.TemporaryFile(dir=self.audit_path.parent) as stream:
                stream.write(b"audit destination probe")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise SessionError("log_unavailable", "Audit log is unavailable.", {
                "path": str(self.audit_path), "error": str(error)}) from error

    def publish_audit(self):
        if self.audit_path is None:
            raise SessionError("log_unavailable", "Set TFT_MCP_AUDIT_PATH to a writable audit file.")
        temporary = None
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            previous = self.audit_path.read_bytes() if self.audit_path.exists() else b""
            with tempfile.NamedTemporaryFile(dir=self.audit_path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(previous)
                for record in self._records:
                    stream.write((json.dumps(record, sort_keys=True) + "\n").encode())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.audit_path)
            temporary = None
        except OSError as error:
            raise SessionError("log_unavailable", "Audit log is unavailable.", {
                "path": str(self.audit_path), "error": str(error)}) from error
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

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

    def prepare_candidate(self):
        if self._candidate_prepared:
            return
        candidate = deepcopy(self.__dict__)
        self.__dict__.update(candidate)
        self._candidate_prepared = True
        if self.game is not None:
            candidate_dir = self.native_root / (self.game_id + "-" + uuid.uuid4().hex)
            accepted_dir = self.native_dir
            self.native_dir = candidate_dir
            shutil.copytree(accepted_dir, candidate_dir)

    @contextmanager
    def lifecycle_transaction(self, mutable=False):
        """Keep candidate fields private under the lock until their audit commit."""
        with SIMULATOR_LOCK:
            if self._records is not None:
                if mutable:
                    self.prepare_candidate()
                yield
                return
            previous = dict(self.__dict__)
            try:
                self._records = []
                if mutable:
                    self.prepare_candidate()
                yield
                self.publish_audit()
                self._records = None
                self._candidate_prepared = False
            except Exception as error:
                diagnostic = self.native_dir if self.native_dir != previous["native_dir"] else None
                failed_id = self.game_id
                self.__dict__.update(previous)
                if isinstance(error, SessionError):
                    if diagnostic is not None:
                        error.details.update(native_log_dir=str(diagnostic), failed_game_id=failed_id)
                    raise
                code = "log_unavailable" if isinstance(error, OSError) else "internal_error"
                raise SessionError(code, "Transaction failed; committed game is unchanged.", {
                    "error": str(error), "native_log_dir": str(diagnostic),
                    "failed_game_id": failed_id}) from error

    def get_game_status(self):
        with SIMULATOR_LOCK:
            return {
                "state": self.state,
                "game_id": self.game_id,
                "controlled_player_id": "player_0" if self.game else None,
                "round": self.game.game_round.current_round if self.game else None,
                "planning_budget": ({"capacity": 14, "remaining": max(0, 14 - self.game.actions_taken["player_0"])}
                                    if self.state == "running" else None),
                "outcome": deepcopy(self.outcome),
            }

    def own_inspection(self, player_id="player_0"):
        if self.game is None:
            raise SessionError("no_game", "Start a game before inspecting player state.")
        if player_id != "player_0":
            raise SessionError("invalid_player", "Only the controlled player is available.", {
                "player_id": player_id, "supported_ids": ["player_0"]})
        if self.state == "terminal":
            if self.terminal_snapshot is None:
                raise SessionError("internal_error", "Terminal own-state snapshot is unavailable.", {
                    "player_id": player_id})
            return deepcopy(self.terminal_snapshot)
        return freeze_player(self.game.player_manager.player_states[player_id], self.game.game_round.current_round)

    def get_board(self, player_id="player_0"):
        with SIMULATOR_LOCK:
            snapshot = self.own_inspection(player_id)
            return {"game_id": self.game_id, "player_id": player_id, "round": snapshot["round"],
                    "slots": [{"location": {"kind": "board", "x": x, "y": y}, "unit": unit}
                              for x, column in enumerate(snapshot["board"]) for y, unit in enumerate(column)],
                    "num_units_in_play": snapshot["num_units_in_play"], "max_units": snapshot["max_units"]}

    def get_bench(self):
        with SIMULATOR_LOCK:
            snapshot = self.own_inspection()
            return {"game_id": self.game_id, "player_id": "player_0", "round": snapshot["round"],
                    "slots": [{"location": {"kind": "bench", "slot": slot}, "unit": unit}
                              for slot, unit in enumerate(snapshot["bench"])]}

    def get_shop(self):
        from Simulator.game.pool_stats import cost_star_values

        with SIMULATOR_LOCK:
            snapshot = self.own_inspection()
            slots = []
            for slot, (offer, unit) in enumerate(zip(snapshot["shop"], snapshot["shop_champions"], strict=True)):
                expected_offer = (f"{unit['champion']}_{unit['chosen']}_c" if unit["chosen"] else unit["champion"]) if unit else None
                if offer != expected_offer:
                    raise SessionError("internal_error", "Stored shop offer is inconsistent.", {
                        "category": "shop", "slot": slot})
                price = cost_star_values[unit["cost"] - 1][unit["stars"] - 1] if unit else None
                slots.append({"slot": slot, "unit": unit, "purchase_cost": price})
            return {"game_id": self.game_id, "player_id": "player_0", "round": snapshot["round"], "slots": slots}

    def get_items(self):
        with SIMULATOR_LOCK:
            snapshot = self.own_inspection()
            return {"game_id": self.game_id, "player_id": "player_0", "round": snapshot["round"],
                    "slots": [{"slot": slot, "item": item} for slot, item in enumerate(snapshot["items"])]}

    def get_economy(self):
        with SIMULATOR_LOCK:
            snapshot = self.own_inspection()
            return {"game_id": self.game_id, "player_id": "player_0", "round": snapshot["round"],
                    **{key: snapshot["economy"][key] for key in ("gold", "health", "level", "exp")},
                    "planning_budget": self.get_game_status()["planning_budget"]}

    def get_traits(self, player_id="player_0"):
        with SIMULATOR_LOCK:
            snapshot = self.own_inspection(player_id)
            traits = snapshot["traits"]
            return {"game_id": self.game_id, "player_id": player_id, "round": snapshot["round"],
                    "traits": [{"trait_id": key, "count": traits["composition"][key], "tier": traits["tiers"][key]}
                               for key in sorted(traits["composition"])]}

    def get_round(self):
        with SIMULATOR_LOCK:
            if self.game is None:
                raise SessionError("no_game", "Start a game before inspecting the round.")
            return {"game_id": self.game_id, "round": self.get_game_status()["round"]}

    @transactional
    def start_game(self, seed):
        if type(seed) is not int or not 0 <= seed <= 2147483647:
            raise SessionError("invalid_input", "seed must be an integer from 0 through 2147483647.")
        if self.state != "idle":
            raise SessionError("game_active", "Close the active game before starting another.", {
                "game_id": self.game_id, "state": self.state,
            })
        game_id = uuid.uuid4().hex
        candidate_dir = self.native_root / game_id if self.native_root else None
        self.probe_audit()
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
                try:
                    with self.simulator_scope(game=candidate, native_dir=candidate_dir):
                        candidate.close()
                except Exception:
                    # Retain the initialization error and unpublished diagnostic directory.
                    pass
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

    @transactional
    def close_game(self):
        outcome = self.outcome
        if self.game is not None:
            if self.state == "running":
                outcome = {"controlled_placement": None, "lobby_complete": False, "reason": "closed_incomplete"}
            self.record("game_closed", game_id=self.game_id, outcome=outcome,
                        native_log_dir=str(self.native_dir))
            with self.simulator_scope():
                self.game.close()
        receipt = {"closed_game_id": self.game_id, "outcome": deepcopy(outcome)}
        self.game = None
        self.game_id = None
        self.state = "idle"
        self.outcome = None
        self.planning_budget = None
        self.placements = {}
        self.terminal_snapshot = None
        self.public_final = {}
        self.baselines = {}
        self.native_dir = None
        self.python_rng = None
        self.baseline_rng = None
        self.module_state = None
        receipt["status"] = self.get_game_status()
        return receipt

    def require_decision(self):
        if self.state == "idle":
            raise SessionError("no_game", "Start a game first.")
        if self.state == "terminal":
            raise SessionError("game_terminal", "The completed game accepts no more actions.")
        if (self.game.agent_selection != "player_0" or self.game.truncations["player_0"]
                or self.game.terminations.get("player_0", True)):
            raise SessionError("internal_error", "No controlled decision is available.")

    def baseline_action(self, player_id):
        player = self.game.player_manager.player_states[player_id]
        mask = self.game.player_manager.action_handlers[player_id].fetch_action_mask()
        encoded = self.baselines[player_id].policy(player, player.shop,
                                                   self.game.game_round.current_round, mask)
        action = [int(part) for part in encoded.split("_")]
        return (action + [0, 0])[:3]

    def step(self, action, kind):
        game = self.game
        player_id = game.agent_selection
        round_number = game.game_round.current_round
        retained = [(key, player) for key, player in game.player_manager.player_states.items()
                    if player is not None and not game.terminations.get(key, True)]
        alive = game.num_alive
        before = (round_number, sum(game.actions_taken.values()), tuple(game.agents))
        game.step(action)
        after = (game.game_round.current_round, sum(game.actions_taken.values()), tuple(game.agents))
        if before == after:
            raise SessionError("internal_error", "Simulator step made no bounded progress.")
        for key, player in retained:
            removed = game.player_manager.player_states.get(key) is None
            if removed and key not in self.placements and player.health <= 0:
                self.placements[key] = alive
                alive -= 1
        for key, player in retained:
            if game.player_manager.player_states.get(key) is None and key not in self.placements:
                self.placements[key] = 1
            if key in self.placements:
                self.public_final[key] = {"health": player.health, "level": player.level}
            if key == "player_0" and key in self.placements and self.terminal_snapshot is None:
                self.terminal_snapshot = freeze_player(player, game.game_round.current_round)
        self.record("progression", game_id=self.game_id, player_id=player_id, round=round_number,
                    kind=kind, action=action, placements=dict(self.placements),
                    native_log_dir=str(self.native_dir))

    def advance(self, old_round=None):
        for _ in range(MAX_PROGRESS_STEPS):
            game = self.game
            if not game.agents or all(game.terminations.values()):
                self.state = "terminal"
                self.outcome = {"controlled_placement": self.placements["player_0"],
                                "lobby_complete": True, "reason": "lobby_complete"}
                self.record("lobby_complete", game_id=self.game_id, outcome=self.outcome,
                            placements=self.placements, native_log_dir=str(self.native_dir))
                return
            selected = game.agent_selection
            if selected == "player_0" and not game.terminations[selected] and not game.truncations[selected]:
                if old_round is None or game.game_round.current_round > old_round:
                    return
            if game.terminations[selected] or game.truncations[selected]:
                self.step(None, "cleanup")
            elif selected == "player_0":
                self.step([0, 0, 0], "controlled_drain")
            else:
                self.step(self.baseline_action(selected), "baseline")
        raise SessionError("internal_error", "Simulator progression exceeded 6000 steps.")

    @transactional
    def controlled_action(self, action):
        """Adapter seam for already-validated singular action tools."""
        self.require_decision()
        if self.get_game_status()["planning_budget"]["remaining"] <= 0:
            raise SessionError("budget_exhausted", "Only end_turn is available until the next round.")
        with self.simulator_scope():
            self.step(action, "controlled_action")
            self.advance()
        return self.get_game_status()

    @transactional
    def end_turn(self):
        self.require_decision()
        old_round = self.game.game_round.current_round
        with self.simulator_scope():
            self.advance(old_round)
        return self.get_game_status()


def freeze_unit(champion):
    """Copy only visible unit fields, without traversing combat references."""
    if champion is None:
        return None
    return {"champion": champion.name, "stars": champion.stars,
            "items": list(champion.items), "chosen": champion.chosen, "cost": champion.cost,
            "kayn_form": getattr(champion, "kayn_form", None),
            "traits": list(champion.origin), "target_dummy": champion.target_dummy,
            "sandguard_overlord_coordinates": deepcopy(getattr(champion, "sandguard_overlord_coordinates", []))}


def freeze_player(player, round_number):
    """Detached post-combat records for later own-state inspection tools."""
    return deepcopy({
        "board": [[freeze_unit(champion) for champion in column] for column in player.board],
        "bench": [freeze_unit(champion) for champion in player.bench],
        "shop": list(player.shop), "shop_champions": [freeze_unit(champion) for champion in player.shop_champions],
        "items": list(player.item_bench),
        "num_units_in_play": player.num_units_in_play, "max_units": player.max_units,
        "economy": {key: getattr(player, key) for key in
                    ("gold", "level", "exp", "health", "win_streak", "loss_streak", "max_units",
                     "refresh_cost", "exp_cost", "level_costs", "max_level")},
        "traits": {"composition": player.team_composition, "tiers": player.team_tiers},
        "round": round_number, "player_round": player.round,
    })


def simulator_identity():
    import Simulator
    from Simulator.simulators.tft_simulator import TFT_Simulator

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
            "environment_name": TFT_Simulator.metadata.get("name"),
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
