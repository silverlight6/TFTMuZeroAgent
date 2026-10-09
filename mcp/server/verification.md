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

## Champion and trait catalog slice

Ticket [#5](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/5), verified on 2026-10-09 in `/tmp/tft-mcp-worktrees/issue-5` on `feat/mcp-5-champions`. Reviewed base is integrated lifecycle commit `de46dd110f6a94229df4af5baa8ea619581138da`. Tested implementation HEAD is `587abcdd8d193f73e58f31e8a7bac639754be85b` with a clean working tree. Merging the current local `feat/mcp-server-main` before final checks reported already up to date. The subsequent verification-record commit changes only this document.

The accepted catalog contract was refreshed against the concrete lifecycle adapter and transport, then recorded in the Spec. Static projections belong to `champion_catalog.py`; transport owns schemas and validation and calls concrete session methods. Imports point into simulator definition tables. No champions are constructed for queries or expected values. No simulator source, root packaging, dependency or CI changes were made.

The source suite passed 22 tests:

```sh
env -u APPIMAGE PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests -q
```

Six catalog tests cover production-launcher SDK discovery and schemas, idle queries, combined filters and empty searches, exact unknown IDs, strict types and unknown keys, Kayn metadata, star gold values, ninja exact activation, fortune parameters, intrinsic trait membership, and running-state status preservation. Focused adapter evidence compares every canonical champion and trait to simulator definitions, snapshots all definition dictionaries/lists and Python/NumPy RNG state, and mutates nested returned values before querying again. Source consistency and defensive-copy checks passed.

The relevant unchanged simulator checks passed 11 tests with the same two PettingZoo observation warnings:

```sh
env -u APPIMAGE /tmp/tft-mcp-env/bin/python -m pytest UnitTests/rng_test.py UnitTests/default_agent_test.py UnitTests/game_round_test.py UnitTests/simulator_test.py -q
```

Complete committed/index/working/untracked scope and whitespace checks passed:

```sh
env -u APPIMAGE /tmp/tft-mcp-env/bin/python mcp/server/scripts/check_scope.py de46dd110f6a94229df4af5baa8ea619581138da
git diff --check de46dd110f6a94229df4af5baa8ea619581138da
```

Red tests first failed because search, champion detail and trait tools were absent. Initial expected literals for Kayn Chosen eligibility and star gold values disagreed with source; those expectations were corrected to exclude tormented and use 5/14/44 gold. No executed check remains failed. This slice used the existing source verification environment without changing installed distributions. Fresh installed-package verification, independent integration review, complete gameplay and actual Codex/Claude host acceptance remain later integration evidence. No release, merge, deployment or owner acceptance is claimed.
