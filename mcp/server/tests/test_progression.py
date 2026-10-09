from tft_mcp.session import GameSession


def test_end_turn_returns_next_real_decision(tmp_path):
    session = GameSession(tmp_path / 'audit.jsonl', tmp_path / 'native')
    assert session.start_game(0)['planning_budget'] == {'capacity': 14, 'remaining': 14}
    result = session.end_turn()
    assert result['round'] == 2
    assert result['planning_budget'] == {'capacity': 14, 'remaining': 14}
    assert session.game.agent_selection == 'player_0'


def test_reserved_budget_never_starts_combat(tmp_path):
    import pytest
    from tft_mcp.session import SessionError
    session = GameSession(tmp_path / 'audit.jsonl', tmp_path / 'native')
    session.start_game(0)
    for _ in range(14):
        session.controlled_action([0, 0, 0])
    assert session.get_game_status()['planning_budget'] == {'capacity': 14, 'remaining': 0}
    assert session.get_game_status()['round'] == 1
    with pytest.raises(SessionError, match='Only end_turn'):
        session.controlled_action([0, 0, 0])
    assert session.end_turn()['round'] == 2


def test_real_seed_zero_elimination_finishes_lobby_and_freezes_state(tmp_path):
    import pytest
    from tft_mcp.session import SessionError
    session = GameSession(tmp_path / 'audit.jsonl', tmp_path / 'native')
    session.start_game(0)
    for _ in range(30):
        status = session.end_turn()
        if status['state'] == 'terminal':
            break
    assert status['state'] == 'terminal'
    assert status['outcome'] == {'controlled_placement': 8, 'lobby_complete': True, 'reason': 'lobby_complete'}
    assert status['planning_budget'] is None
    assert status['round'] > session.terminal_snapshot['round']
    assert session.terminal_snapshot['economy']['health'] <= 0
    assert set(session.terminal_snapshot) == {'board', 'bench', 'shop', 'shop_champions', 'items', 'economy', 'traits', 'round', 'player_round', 'num_units_in_play', 'max_units'}
    frozen = session.terminal_snapshot.copy()
    with pytest.raises(SessionError) as error:
        session.end_turn()
    assert error.value.code == 'game_terminal'
    with pytest.raises(SessionError) as error:
        session.start_game(1)
    assert error.value.code == 'game_active'
    status['outcome']['controlled_placement'] = 1
    assert session.get_game_status()['outcome']['controlled_placement'] == 8
    assert session.terminal_snapshot == frozen
    assert session.close_game()['outcome']['controlled_placement'] == 8
    assert session.start_game(0)['round'] == 1


def gameplay(session):
    from tft_mcp.session import freeze_player
    return {'status': {key: value for key, value in session.get_game_status().items() if key != 'game_id'},
            'players': {key: freeze_player(player, session.game.game_round.current_round)
                        for key, player in session.game.player_manager.player_states.items() if player is not None},
            'placements': session.placements, 'snapshot': session.terminal_snapshot,
            'episode_rng': repr(session.game.rng.py.getstate()),
            'baseline_rng': repr(session.baseline_rng)}


def test_failed_candidate_restores_graph_logs_rng_and_retries(tmp_path, monkeypatch):
    import builtins
    import os
    import random
    import numpy as np
    import pytest
    from Simulator.simulators.tft_simulator import TFT_Simulator
    from Simulator.battle import champion, origin_class
    from tft_mcp.session import SessionError
    for failure in ('step', 'native', 'audit'):
        session = GameSession(tmp_path / failure / 'audit.jsonl', tmp_path / failure / 'native')
        reference = GameSession(tmp_path / (failure + '-reference') / 'audit.jsonl', tmp_path / (failure + '-reference') / 'native')
        session.start_game(0)
        reference.start_game(0)
        original_game = session.game
        original_policies = session.baselines
        original_baseline_rng = session.baseline_rng
        original_modules = session.module_state
        expected = gameplay(session)
        audit = session.audit_path.read_bytes()
        native = (session.native_dir / 'log.txt').read_bytes()
        rng = random.getstate()
        numpy_rng = np.random.get_state()
        bindings = (champion.log, champion.test_multiple, origin_class.game_compositions, origin_class.game_comp_tiers)
        with monkeypatch.context() as patch:
            if failure == 'step':
                original = TFT_Simulator.step
                count = 0
                def fail_step(game, action):
                    nonlocal count
                    result = original(game, action)
                    count += 1
                    if count == 12:
                        raise RuntimeError('injected after policies, encoders, pool and RNG mutate')
                    return result
                patch.setattr(TFT_Simulator, 'step', fail_step)
            elif failure == 'native':
                original = builtins.open
                def fail_open(path, *args, **kwargs):
                    if str(path) == 'log.txt':
                        with original(path, *args, **kwargs) as stream:
                            stream.write('partial candidate native write')
                        raise OSError('injected native partial write')
                    return original(path, *args, **kwargs)
                patch.setattr(builtins, 'open', fail_open)
            else:
                def fail_replace(*args):
                    raise OSError('injected audit replacement')
                patch.setattr(os, 'replace', fail_replace)
            with pytest.raises(SessionError):
                session.end_turn()
        assert session.game is original_game
        assert session.baselines is original_policies
        assert session.baseline_rng is original_baseline_rng
        assert session.module_state is original_modules
        assert gameplay(session) == expected
        assert session.audit_path.read_bytes() == audit
        assert (session.native_dir / 'log.txt').read_bytes() == native
        assert random.getstate() == rng
        assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
        assert np.random.get_state()[2:] == numpy_rng[2:]
        assert all(actual is expected for actual, expected in zip(
            (champion.log, champion.test_multiple, origin_class.game_compositions, origin_class.game_comp_tiers), bindings))
        session.end_turn()
        reference.end_turn()
        assert gameplay(session) == gameplay(reference)
        assert session.game.rng is session.game.combat_ctx.rng
        assert session.game.pool_obj is session.game.player_manager.pool_obj
        assert session.game.game_round.PLAYERS is session.game.player_manager.player_states
        manager = session.game.player_manager
        assert session.game.step_function.player_manager is manager
        for key, player in manager.player_states.items():
            if player is not None:
                assert manager.observation_states[key].player is player
                assert manager.action_handlers[key].player is player
                assert player.pool_obj is session.game.pool_obj


def test_late_transport_result_failure_and_failed_close_keep_committed_game(tmp_path, monkeypatch):
    import os
    import pytest
    from Simulator.simulators.tft_simulator import TFT_Simulator
    from tft_mcp.session import SessionError
    session = GameSession(tmp_path / 'audit.jsonl', tmp_path / 'native')
    session.start_game(0)
    original_game = session.game
    audit = session.audit_path.read_bytes()
    with monkeypatch.context() as patch:
        def fail_replace(*args):
            raise OSError('injected tool result commit failure')
        patch.setattr(os, 'replace', fail_replace)
        with pytest.raises(SessionError):
            with session.lifecycle_transaction():
                session.record('tool_request', tool='end_turn')
                session.end_turn()
                session.record('tool_result', tool='end_turn')
        with pytest.raises(SessionError):
            session.close_game()
    assert session.game is original_game
    assert session.audit_path.read_bytes() == audit
    with monkeypatch.context() as patch:
        def fail_close(game):
            game.player_manager.player_states['player_0'].gold = 1000
            raise RuntimeError('injected close mutation')
        patch.setattr(TFT_Simulator, 'close', fail_close)
        with pytest.raises(SessionError):
            session.close_game()
    assert session.game is original_game
    assert session.audit_path.read_bytes() == audit
    assert session.close_game()['status']['state'] == 'idle'


def test_progress_bound_discards_candidate_and_allows_real_retry(tmp_path, monkeypatch):
    import pytest
    import tft_mcp.session as module
    session = GameSession(tmp_path / 'audit.jsonl', tmp_path / 'native')
    session.start_game(0)
    expected = gameplay(session)
    audit = session.audit_path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(module, 'MAX_PROGRESS_STEPS', 1)
        with pytest.raises(module.SessionError, match='exceeded'):
            session.end_turn()
    assert gameplay(session) == expected
    assert session.audit_path.read_bytes() == audit
    assert session.end_turn()['round'] == 2


def test_unavailable_audit_is_rejected_before_native_construction(tmp_path):
    import pytest
    from tft_mcp.session import SessionError
    audit = tmp_path / 'audit.jsonl'
    audit.mkdir()
    native_root = tmp_path / 'native'
    session = GameSession(audit, native_root)
    with pytest.raises(SessionError) as error:
        session.start_game(0)
    assert error.value.code == 'log_unavailable'
    assert session.get_game_status()['state'] == 'idle'
    assert not native_root.exists()


def test_recorded_status_reads_preserve_committed_graph_and_native_directory(tmp_path):
    session = GameSession(tmp_path / 'audit.jsonl', tmp_path / 'native')
    status = session.start_game(0)
    game = session.game
    native = session.native_dir
    rng = session.baseline_rng
    with session.lifecycle_transaction():
        session.record('tool_request', tool='get_game_status')
        assert session.get_game_status() == status
        session.record('tool_result', tool='get_game_status', result=status)
    assert session.game is game
    assert session.native_dir is native
    assert session.baseline_rng is rng


def test_native_winner_cleanup_retains_controlled_snapshot(tmp_path):
    session = GameSession(tmp_path / 'audit.jsonl', tmp_path / 'native')
    session.start_game(0)
    # A focused native-player fixture exercises survivor removal, separate from seed-zero acceptance.
    for key, player in session.game.player_manager.player_states.items():
        if key != 'player_0':
            player.health = -1000
    status = session.end_turn()
    assert status['state'] == 'terminal'
    assert status['outcome']['controlled_placement'] == 1
    assert session.terminal_snapshot['economy']['health'] > 0
    assert session.public_final['player_0']['health'] == session.terminal_snapshot['economy']['health']
    assert sorted(session.placements.values()) == list(range(1, 9))
