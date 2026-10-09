# Local verification

Ticket [#2](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/2), verified on 2026-10-09. Reviewed base and precommit HEAD were `54cbb8bb9e9933a80ea04bf99989bf001fe4774e` on `feat/mcp-2-lifecycle`. `feat/mcp-server-main` was merged before handoff and reported already up to date. The tested working diff contains only `mcp/` documentation, server packaging, source, tests and local tooling. The commit containing this record owns that tested implementation. Source, tests, scripts and `pyproject.toml` have combined SHA-256 `f4b1c2dcbe6f6c753c45a69af2d05ad62061ff18129c35f29c0471c97c03b664`.

The accepted contract was reviewed against the unchanged real simulator before implementation, as recorded in the [Spec](../SPEC.md#lifecycle-slice-source-review). Root review identified incomplete Git-layer scope checking; regression tests first reproduced staged restoration and rename escapes, then passed after checking all layers without rename folding. A committed outside change restored in the index also has coverage.

The source test command passed 16 tests:

```sh
env -u APPIMAGE PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests -q
```

Protocol evidence covers official SDK initialization and tool discovery, idle status, real eight-player construction, strict seeds and unknown arguments, active-game rejection, close and restart, startup audit/native failures, failed close retaining the game, recorded interpreter hash after random/42/0 launches, ordered logging, runtime versions, clean JSON protocol stdout and native malformed-envelope errors. Focused real-simulator adapter evidence covers global Python/NumPy RNG isolation, module-binding preservation, failed initialization, retained failed-start diagnostics, startup and late audit-durability failures and cleanup. Scope tests use temporary real Git repositories.

Fresh CPU-only noneditable installation used Python 3.14.7, MCP 1.30.0, NumPy 2.5.3, PettingZoo 1.27.0 and Gymnasium 1.4.0. No torch, TensorFlow or JAX package was installed. The simulator was archived from the fixed base into `/tmp/tft-mcp-install-source-2`, then installed before the extension:

```sh
uv venv /tmp/tft-mcp-install-2
uv pip install --python /tmp/tft-mcp-install-2/bin/python /tmp/tft-mcp-install-source-2
uv pip install --python /tmp/tft-mcp-install-2/bin/python '/tmp/tft-mcp-install-source-2/mcp/server-ready[dev]'
```

`server-ready` contains the final extension source copied from this slice. After source updates, that separate extension distribution was reinstalled with `--reinstall-package tft-mcp-server`. Both distribution `direct_url.json` files have empty `dir_info`, with no editable flag. Outside the checkout, simulator and extension imports resolve to `/tmp/tft-mcp-install-2/lib64/python3.14/site-packages/`.

From `/tmp`, the absolute installed console launcher passed all six protocol tests without `PYTHONPATH` source imports:

```sh
env -u APPIMAGE TFT_MCP_TEST_COMMAND=/tmp/tft-mcp-install-2/bin/tft-mcp /tmp/tft-mcp-install-2/bin/python -m pytest -c /tmp/tft-mcp-worktrees/issue-2/mcp/server/pyproject.toml /tmp/tft-mcp-worktrees/issue-2/mcp/server/tests/test_protocol.py -q
```

The documented `env -u APPIMAGE` launch resolves an observed host problem: inherited `APPIMAGE` makes Python report the T3 Code AppImage executable and omit virtual-environment packages. The production installed launcher works from outside the checkout after removal of that variable, including interpreter re-execution. This is an actual installed entry-point test, not a source-only substitute.

Relevant unchanged simulator checks passed 11 tests with two existing PettingZoo observation warnings:

```sh
env -u APPIMAGE PYTHONHASHSEED=0 /tmp/tft-mcp-env/bin/python -m pytest UnitTests/rng_test.py UnitTests/default_agent_test.py UnitTests/game_round_test.py UnitTests/simulator_test.py -q
```

The complete scope check and whitespace check passed:

```sh
env -u APPIMAGE /tmp/tft-mcp-env/bin/python mcp/server/scripts/check_scope.py 54cbb8bb9e9933a80ea04bf99989bf001fe4774e
git diff --check
```

Scope includes committed HEAD, index, working tree and untracked paths; renamed source paths are checked separately. Simulator source, root packaging, mandatory dependencies and CI files are unchanged. Failure-path filesystem checks use actual `/proc` write failures, not mode-bit assumptions that privileged execution could bypass.

Initial red checks failed as intended for missing lifecycle implementation, missing production launcher and missing actual interpreter-hash recording. Those failures are resolved. No remaining executed check failed. Scheduling, executable terminal transitions, complete-game replay, later action tools and real Codex/Claude client acceptance were not executed for this bounded lifecycle slice; they remain owned by their later tickets. No release, PR merge, owner acceptance or deployment is claimed here.

## Ticket #6 item catalog verification

Reviewed base and refreshed concrete lifecycle seam: `de46dd110f6a94229df4af5baa8ea619581138da`. Implementation head tested: `f1b24b17e4f71bd11bcf5a3f8dc8404355b83f2d` on `feat/mcp-6-items` in `/tmp/tft-mcp-worktrees/issue-6`. The integration tip was merged before final checks and was already the base. The final documentation-only commit adds this record. Working changes were clean before the record.

Passed commands, using the existing CPU-only environment without changing its installation:

```sh
env -u APPIMAGE PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests -q
env -u APPIMAGE PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest UnitTests/rng_test.py UnitTests/default_agent_test.py UnitTests/game_round_test.py UnitTests/simulator_test.py -q
env -u APPIMAGE /tmp/tft-mcp-env/bin/python mcp/server/scripts/check_scope.py de46dd110f6a94229df4af5baa8ea619581138da
git diff --check de46dd110f6a94229df4af5baa8ea619581138da
```

The extension suite passed 22 tests. The affected simulator checks passed 11 tests with two existing PettingZoo observation warnings. Scope passed for eight changed paths before this record; all implementation and tests live under `mcp/server/`, with shared contract changes only in `mcp/SPEC.md`. No simulator files, root packaging, dependency definitions, or CI were changed.

TDD recorded failing adapter search, adapter detail and production discovery tests before their corresponding implementation. Official MCP SDK clients launched the production module outside the checkout, exercised strict schemas and structured errors, fetched every canonical item while idle, and queried a running game. Focused real-session checks compared source definitions, Python/NumPy/session RNG, player shops, inventories, gold, health, boards and benches before and after repeated catalogs. Nested returned data were mutated to prove defensive copies. Representative source constraints cover all five consumables and Thieves' Gloves; recipes cover duplicate ingredients and granted traits.

The first broad source snapshot check failed because it included Python module `__builtins__`; the check now captures definition tables only. Subsequent checks passed. The static helper also excludes module metadata from effect tables.

Fresh installation, real Codex connection, complete gameplay and working Kayn combat transformation were not executed by this slice. Existing installation acceptance belongs to lifecycle/setup; complete gameplay belongs to later slices. Kayn's source mismatch is explicitly exposed in catalog constraints and recorded in the Spec, with no core fix or promise of working transformation. Technical slice verification does not establish host-client or owner acceptance.
