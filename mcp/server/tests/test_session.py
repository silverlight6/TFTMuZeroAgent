from pathlib import Path
from tft_mcp.session import GameSession


def test_idle_status_and_repeated_close(tmp_path):
    session = GameSession(tmp_path / "audit.jsonl", tmp_path / "native")
    idle = {
        "state": "idle", "game_id": None, "controlled_player_id": None,
        "round": None, "planning_budget": None, "outcome": None,
    }
    assert session.get_game_status() == idle
    assert session.close_game() == {"closed_game_id": None, "outcome": None, "status": idle}
    assert session.close_game()["status"] == idle


def test_real_seeded_game_lifecycle_and_isolation(tmp_path):
    import random
    import numpy as np
    from tft_mcp.session import SessionError

    session = GameSession(tmp_path / "audit.jsonl", tmp_path / "native")
    random.seed(91)
    np.random.seed(92)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    started = session.start_game(123)
    assert started["state"] == "running"
    assert started["controlled_player_id"] == "player_0"
    assert started["round"] == 1
    assert len(session.baselines) == 7
    assert set(session.game.player_manager.player_states) == {f"player_{i}" for i in range(8)}
    shop = list(session.game.player_manager.player_states["player_0"].shop)
    try:
        session.start_game(124)
    except SessionError as error:
        assert error.code == "game_active"
    else:
        raise AssertionError("An active game must reject another start")
    assert session.get_game_status() == started
    receipt = session.close_game()
    assert receipt["closed_game_id"] == started["game_id"]
    assert receipt["outcome"] == {"controlled_placement": None, "lobby_complete": False, "reason": "closed_incomplete"}
    assert session.get_game_status()["state"] == "idle"
    assert session.start_game(123)["game_id"] != started["game_id"]
    assert session.game.player_manager.player_states["player_0"].shop == shop
    session.close_game()
    assert random.getstate() == python_state
    restored = np.random.get_state()
    assert restored[0] == numpy_state[0]
    assert np.array_equal(restored[1], numpy_state[1])
    assert restored[2:] == numpy_state[2:]


def test_failed_initialization_restores_rng_and_retains_diagnostic_evidence(tmp_path, monkeypatch):
    import builtins
    import random
    import numpy as np
    import pytest
    from tft_mcp.session import SessionError

    original_open = builtins.open
    def failed_native_write(path, *args, **kwargs):
        if str(path) == "log.txt":
            raise RuntimeError("Injected native filesystem failure")
        return original_open(path, *args, **kwargs)

    session = GameSession(tmp_path / "audit.jsonl", tmp_path / "native")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", failed_native_write)
        with pytest.raises(SessionError) as failed:
            session.start_game(123)
    assert failed.value.code == "internal_error"
    assert session.get_game_status()["state"] == "idle"
    assert not session.baselines
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    evidence = Path(failed.value.details["native_log_dir"])
    assert evidence.is_dir()
    # Failed candidate records are unpublished; the error identifies native evidence.
    assert not (tmp_path / "audit.jsonl").exists()
    assert session.start_game(123)["state"] == "running"
    session.close_game()


def test_session_preserves_other_simulator_module_state(tmp_path):
    from copy import deepcopy
    from Simulator.battle import champion, origin_class

    bindings = (champion.log, champion.test_multiple,
                origin_class.game_compositions, origin_class.game_comp_tiers)
    compositions = deepcopy(origin_class.game_compositions)
    tiers = deepcopy(origin_class.game_comp_tiers)
    diagnostics = deepcopy(champion.test_multiple)
    session = GameSession(tmp_path / "audit.jsonl", tmp_path / "native")
    session.start_game(123)
    session.close_game()
    assert champion.log is bindings[0]
    assert champion.test_multiple is bindings[1]
    assert origin_class.game_compositions is bindings[2]
    assert origin_class.game_comp_tiers is bindings[3]
    assert origin_class.game_compositions == compositions
    assert origin_class.game_comp_tiers == tiers
    assert champion.test_multiple == diagnostics


def test_final_audit_failure_rolls_back_started_game(tmp_path, monkeypatch):
    import os
    import pytest
    from tft_mcp.session import SessionError

    session = GameSession(tmp_path / "audit.jsonl", tmp_path / "native")
    original_fsync = os.fsync
    with pytest.raises(SessionError) as failed:
        with session.lifecycle_transaction():
            session.start_game(123)
            def fail_fsync(descriptor):
                raise OSError("Injected audit durability failure")
            monkeypatch.setattr(os, "fsync", fail_fsync)
            session.record("tool_result", tool="start_game", is_error=False)
    monkeypatch.setattr(os, "fsync", original_fsync)
    assert failed.value.code == "log_unavailable"
    assert failed.value.details["failed_game_id"]
    assert Path(failed.value.details["native_log_dir"]).is_dir()
    assert session.get_game_status()["state"] == "idle"
    assert not session.baselines
    assert session.start_game(123)["state"] == "running"
    session.close_game()


def test_startup_audit_durability_failure_identifies_unpublished_diagnostics(tmp_path, monkeypatch):
    import os
    import pytest
    from tft_mcp.session import SessionError

    session = GameSession(tmp_path / "audit.jsonl", tmp_path / "native")
    original_fsync = os.fsync
    writes = 0
    def fail_after_construction(descriptor):
        nonlocal writes
        writes += 1
        if writes >= 3:
            raise OSError("Injected startup audit durability failure")
        return original_fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_after_construction)
        with pytest.raises(SessionError) as failed:
            session.start_game(123)
    assert failed.value.code == "log_unavailable"
    assert failed.value.details["failed_game_id"]
    assert Path(failed.value.details["native_log_dir"]).is_dir()
    assert session.get_game_status()["state"] == "idle"
    assert not session.baselines
    assert session.start_game(123)["state"] == "running"
    session.close_game()
