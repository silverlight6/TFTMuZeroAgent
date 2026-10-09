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

    def public_player(self, player_id, category):
        if self.game is None:
            raise SessionError("no_game", "Start a game before inspecting player state.")
        manager = self.game.player_manager
        if player_id not in manager.player_ids:
            raise SessionError("invalid_player", "Select a player_id from get_players.", {
                "player_id": player_id, "supported_ids": sorted(manager.player_ids)})
        player = manager.player_states.get(player_id)
        if player is None:
            raise SessionError("player_eliminated", f"This player's {category} is unavailable after removal. Use get_players for final public status.", {
                "player_id": player_id, "category": category})
        return player

    def get_players(self):
        with SIMULATOR_LOCK:
            if self.game is None:
                raise SessionError("no_game", "Start a game before inspecting players.")
            manager = self.game.player_manager
            players = []
            for player_id in sorted(manager.player_ids):
                player = manager.player_states.get(player_id)
                placement = self.placements.get(player_id)
                public = ({"health": player.health, "level": player.level} if player is not None
                          else self.public_final.get(player_id, {}))
                status = "alive"
                if placement == 1:
                    status = "winner"
                elif placement is not None or player is None:
                    status = "eliminated"
                players.append({"player_id": player_id, "controlled": player_id == "player_0",
                                "status": status,
                                "health": public.get("health"), "level": public.get("level"), "placement": placement})
            return {"game_id": self.game_id, "players": players}

    def get_board(self, player_id="player_0"):
        with SIMULATOR_LOCK:
            if player_id == "player_0":
                snapshot = self.own_inspection()
            else:
                player = self.public_player(player_id, "board")
                snapshot = {"board": [[freeze_unit(unit) for unit in column] for column in player.board],
                            "num_units_in_play": player.num_units_in_play, "max_units": player.max_units,
                            "round": self.game.game_round.current_round}
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
            if player_id == "player_0":
                snapshot = self.own_inspection()
                traits = snapshot["traits"]
                round_number = snapshot["round"]
            else:
                player = self.public_player(player_id, "traits")
                traits = {"composition": player.team_composition, "tiers": player.team_tiers}
                round_number = self.game.game_round.current_round
            return {"game_id": self.game_id, "player_id": player_id, "round": round_number,
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

    def require_action_budget(self):
        self.require_decision()
        if self.get_game_status()["planning_budget"]["remaining"] <= 0:
            raise SessionError("budget_exhausted", "Only end_turn is available until the next round.")

    @transactional
    def refresh_shop(self, **arguments):
        if arguments:
            raise SessionError("invalid_input", "refresh_shop takes no arguments.")
        self.require_action_budget()
        player = self.game.player_manager.player_states["player_0"]
        cost, gold = player.refresh_cost, player.gold
        if gold < cost:
            raise SessionError("insufficient_gold", "Shop refresh costs more gold than is available.",
                               {"resource": "gold", "required": cost, "available": gold})
        shop, champions = player.shop, player.shop_champions
        before_status = self.get_game_status()
        status = self.controlled_action([2, 0, 0])
        if player.gold != gold - cost or player.shop is shop or player.shop_champions is champions:
            raise SessionError("internal_error", "Shop refresh postconditions failed.")
        check_action_status(before_status, status, self.game.agent_selection)
        slots = self.get_shop()["slots"]
        if len(slots) != 5:
            raise SessionError("internal_error", "Shop refresh did not produce five offers.")
        return {"gold_spent": cost, "slots": slots, "status": status}

    @transactional
    def buy_xp(self, **arguments):
        if arguments:
            raise SessionError("invalid_input", "buy_xp takes no arguments.")
        self.require_action_budget()
        player = self.game.player_manager.player_states["player_0"]
        level, xp, capacity = player.level, player.exp, player.max_units
        cap, cost, gold = player.max_level, player.exp_cost, player.gold
        thresholds = list(player.level_costs)
        if level >= cap:
            raise SessionError("level_cap", "The player has reached the native level cap.",
                               {"level": level, "max_level": cap})
        if gold < cost:
            raise SessionError("insufficient_gold", "Experience costs more gold than is available.",
                               {"resource": "gold", "required": cost, "available": gold})
        before_status = self.get_game_status()
        status = self.controlled_action([1, 0, 0])
        if not level <= player.level <= cap:
            raise SessionError("internal_error", "Experience level postcondition failed.")
        spent_xp = sum(thresholds[level:player.level])
        available_xp = xp + cost
        valid_xp = (player.exp == 0 and spent_xp <= available_xp if player.level == cap else
                    player.exp == available_xp - spent_xp and 0 <= player.exp < thresholds[player.level])
        if player.gold != gold - cost or player.max_units != capacity + player.level - level or not valid_xp:
            raise SessionError("internal_error", "Experience postconditions failed.")
        check_action_status(before_status, status, self.game.agent_selection)
        return {"gold_spent": cost, "xp_before": xp, "xp": player.exp,
                "level_before": level, "level": player.level, "unit_capacity": player.max_units, "status": status}

    @transactional
    def buy_unit(self, **arguments):
        validate_buy_arguments(arguments)
        self.require_action_budget()
        slot = arguments["shop_slot"]
        player = self.game.player_manager.player_states["player_0"]
        unit = player.shop_champions[slot]
        if unit is None:
            if player.shop[slot] is not None:
                raise SessionError("internal_error", "Stored shop offer is inconsistent.", {"shop_slot": slot})
            raise SessionError("empty_slot", "Select an occupied shop slot.", {"shop_slot": slot})
        purchased = freeze_unit(unit)
        expected_offer = f"{unit.name}_{unit.chosen}_c" if unit.chosen else unit.name
        if player.shop[slot] != expected_offer:
            raise SessionError("internal_error", "Stored shop offer is inconsistent.", {"shop_slot": slot})
        price = action_price(unit)
        if player.gold < price:
            raise SessionError("insufficient_gold", "The offer costs more gold than is available.",
                               {"shop_slot": slot, "required": price, "available": player.gold, "resource": "gold"})
        expected_items = merge_inventory(player, unit)
        if all(player.bench):
            entry = catalog_entry(player, unit.name, unit.stars)
            if entry is None or entry["num"] != 2:
                raise SessionError("capacity_exceeded", "The bench is full and this offer cannot merge.",
                                   {"shop_slot": slot, "resource": "bench", "required": 1, "available": 0})
        before = action_snapshot(player)
        status = self.controlled_action([3, slot, 0])
        after = action_snapshot(player)
        if player.shop[slot] is not None or player.shop_champions[slot] is not None or player.gold != before["gold"] - price or after["items"] != expected_items:
            raise SessionError("internal_error", "Purchase postconditions failed.", {"shop_slot": slot})
        check_copy_change(before, after, purchased, 1)
        return {"shop_slot": slot, "purchased": purchased, "gold_spent": price,
                "unit_changes": unit_changes(before, after), "item_changes": item_changes(before, after), "status": status}

    @transactional
    def sell_unit(self, **arguments):
        validate_sell_arguments(arguments)
        self.require_action_budget()
        location = arguments["location"]
        player = self.game.player_manager.player_states["player_0"]
        unit = (player.board[location["x"]][location["y"]] if location["kind"] == "board"
                else player.bench[location["slot"]])
        if unit is None:
            raise SessionError("empty_slot", "Select an occupied owned location.", {"location": location})
        sold = freeze_unit(unit)
        price = action_price(unit)
        entry = catalog_entry(player, unit.name, unit.stars)
        count = sum(1 for column in player.board for owned in column if owned and owned.name == unit.name and owned.stars == unit.stars)
        count += sum(1 for owned in player.bench if owned and owned.name == unit.name and owned.stars == unit.stars)
        if entry is None or entry["num"] != count:
            raise SessionError("internal_error", "Owned units do not match the native catalog.", {"location": location})
        equipment = real_items(unit)
        available = player.item_bench.count(None)
        if location["kind"] == "board" and len(equipment) > available:
            raise SessionError("capacity_exceeded", "Board equipment cannot fit in inventory.",
                               {"location": location, "resource": "items", "required": len(equipment), "available": available})
        returned = equipment if len(equipment) <= available else []
        dropped = [] if returned == equipment else equipment
        before = action_snapshot(player)
        flat = location["x"] * 4 + location["y"] if location["kind"] == "board" else 28 + location["slot"]
        status = self.controlled_action([4, flat, 0])
        after = action_snapshot(player)
        remaining = (player.board[location["x"]][location["y"]] if location["kind"] == "board"
                     else player.bench[location["slot"]])
        expected_items = list(before["items"])
        for item in returned:
            expected_items[expected_items.index(None)] = item
        if remaining is not None or player.gold != before["gold"] + price or after["items"] != expected_items:
            raise SessionError("internal_error", "Sale postconditions failed.", {"location": location})
        check_copy_change(before, after, sold, -1)
        if location["kind"] == "board" and sold["champion"] == "azir":
            if any(player.board[x][y] is not None for x, y in sold["sandguard_overlord_coordinates"]):
                raise SessionError("internal_error", "Azir sandguard removal failed.", {"location": location})
        return {"location": deepcopy(location), "sold": sold, "gold_gained": price,
                "returned_items": returned, "dropped_items": dropped,
                "unit_changes": unit_changes(before, after), "status": status}

    @transactional
    def end_turn(self):
        self.require_decision()
        old_round = self.game.game_round.current_round
        with self.simulator_scope():
            self.advance(old_round)
        return self.get_game_status()


def validate_buy_arguments(arguments):
    if set(arguments) != {"shop_slot"} or type(arguments.get("shop_slot")) is not int or not 0 <= arguments["shop_slot"] <= 4:
        raise SessionError("invalid_input", "buy_unit requires only shop_slot, an integer from 0 through 4.", {"field": "shop_slot"})


def validate_sell_arguments(arguments):
    location = arguments.get("location")
    if set(arguments) != {"location"} or type(location) is not dict:
        raise SessionError("invalid_input", "sell_unit requires only a board or bench location.", {"field": "location"})
    kind = location.get("kind")
    fields = {"kind", "x", "y"} if kind == "board" else {"kind", "slot"} if kind == "bench" else set()
    bounds = {"x": 6, "y": 3} if kind == "board" else {"slot": 8}
    if not fields or set(location) != fields or any(type(location.get(key)) is not int or not 0 <= location[key] <= bound for key, bound in bounds.items()):
        raise SessionError("invalid_input", "Use exact board x=0..6,y=0..3 or bench slot=0..8 coordinates.", {"field": "location"})


def action_price(unit):
    from Simulator.battle.stats import BASE_CHAMPION_LIST, COST
    from Simulator.game.pool_stats import cost_star_values
    if unit.target_dummy or unit.name == "sandguard":
        raise SessionError("unsupported_action", "Dummy units cannot be purchased or sold.", {"reason": "dummy_unit"})
    if unit.name not in BASE_CHAMPION_LIST:
        raise SessionError("internal_error", "Unit is absent from the installed champion catalog.", {"champion": unit.name})
    if type(unit.cost) is not int or type(unit.stars) is not int or not 1 <= unit.cost <= 5 or not 1 <= unit.stars <= 3:
        raise SessionError("unsupported_action", "Unit has no supported native action price.", {"reason": "price_range", "champion": unit.name})
    if unit.cost != COST[unit.name]:
        raise SessionError("internal_error", "Unit cost differs from the installed champion definition.", {"champion": unit.name})
    return cost_star_values[unit.cost - 1][unit.stars - 1]


def catalog_entry(player, name, stars):
    entries = [entry for entry in player.triple_catalog if entry["name"] == name and entry["level"] == stars]
    if len(entries) > 1 or (entries and (type(entries[0]["num"]) is not int or not 1 <= entries[0]["num"] <= 2)):
        raise SessionError("internal_error", "Native triple catalog is inconsistent.", {"champion": name, "stars": stars})
    return entries[0] if entries else None


def real_items(unit):
    return ["thieves_gloves"] if unit.items and unit.items[0] == "thieves_gloves" else list(unit.items)


def merge_inventory(player, incoming):
    """Check native bench returns before board returns across reachable merges."""
    inventory = list(player.item_bench)
    stars, chosen = incoming.stars, incoming.chosen
    visited = set()
    while True:
        if stars in visited or not 1 <= stars <= 3:
            raise SessionError("unsupported_action", "Native promotion is outside supported action stars.",
                               {"reason": "promotion_range", "champion": incoming.name})
        visited.add(stars)
        entry = catalog_entry(player, incoming.name, stars)
        bench = [unit for unit in player.bench if unit and unit.name == incoming.name and unit.stars == stars]
        board = [unit for column in player.board for unit in column if unit and unit.name == incoming.name and unit.stars == stars]
        contributors = bench + board
        if len(contributors) != (entry["num"] if entry else 0):
            raise SessionError("internal_error", "Native catalog does not match owned contributors.",
                               {"champion": incoming.name, "stars": stars})
        if entry is None or entry["num"] != 2:
            return inventory
        for unit in contributors:
            equipment = real_items(unit)
            available = inventory.count(None)
            if len(equipment) > available:
                if unit in board:
                    raise SessionError("capacity_exceeded", "Merge board equipment cannot fit in inventory.",
                                       {"resource": "items", "required": len(equipment), "available": available,
                                        "location": {"kind": "board", "x": unit.x, "y": unit.y}})
                continue
            for item in equipment:
                inventory[inventory.index(None)] = item
        promoted = 3 if chosen else stars + 1
        chosen = False
        for unit in contributors:
            if unit.chosen:
                chosen = unit.chosen
        if promoted <= stars:
            raise SessionError("unsupported_action", "Native promotion does not progress.", {"reason": "promotion_cycle"})
        stars = promoted


def copy_weights(snapshot):
    weights = {}
    for _, unit in snapshot["units"]:
        if unit and not unit["target_dummy"] and unit["champion"] != "sandguard":
            weights[unit["champion"]] = weights.get(unit["champion"], 0) + 3 ** (unit["stars"] - 1)
    return weights


def check_copy_change(before, after, unit, direction):
    expected = copy_weights(before)
    name = unit["champion"]
    expected[name] = expected.get(name, 0) + direction * 3 ** (unit["stars"] - 1)
    expected = {name: count for name, count in expected.items() if count}
    if copy_weights(after) != expected:
        raise SessionError("internal_error", "Native action did not preserve expected owned copies.", {"champion": name})


def action_snapshot(player):
    return {"units": [({"kind": "board", "x": x, "y": y}, freeze_unit(unit))
                      for x, column in enumerate(player.board) for y, unit in enumerate(column)] +
                     [({"kind": "bench", "slot": slot}, freeze_unit(unit)) for slot, unit in enumerate(player.bench)],
            "items": list(player.item_bench), "gold": player.gold}


def unit_changes(before, after):
    return [{"location": location, "before": old, "after": new}
            for (location, old), (_, new) in zip(before["units"], after["units"], strict=True) if old != new]


def item_changes(before, after):
    return [{"slot": slot, "before": old, "after": new}
            for slot, (old, new) in enumerate(zip(before["items"], after["items"], strict=True)) if old != new]


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


def check_action_status(before, after, selection):
    if (after["state"] != "running" or after["round"] != before["round"] or selection != "player_0"
            or after["planning_budget"]["remaining"] != before["planning_budget"]["remaining"] - 1):
        raise SessionError("internal_error", "Singular action changed the decision boundary.")
