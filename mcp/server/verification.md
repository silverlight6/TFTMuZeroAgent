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

## Ticket #5 refresh after item catalog integration

The isolated `feat/mcp-5-champions` branch merged verified integration tip `24f8fb7a5498126f7cb608b61079b4c65edb88a6` before handoff. Tested merge HEAD is `8155faade1ff4eb5918811729316e54c0b008b62`, with a clean working tree. Additive conflicts in session methods, transport registration/validation/dispatch, README and verification records preserve both catalog contracts. The lifecycle discovery test retains the integrated subset assertion. The Spec merged both accepted contracts. No unintegrated progression code was included.

The combined source suite passed 28 tests, including both item and champion/trait production MCP journeys and focused adapter checks:

```sh
env -u APPIMAGE PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests -q
env -u APPIMAGE /tmp/tft-mcp-env/bin/python mcp/server/scripts/check_scope.py 24f8fb7a5498126f7cb608b61079b4c65edb88a6
git diff --check 24f8fb7a5498126f7cb608b61079b4c65edb88a6
```

Scope passed for seven changed paths against the refreshed base. The whitespace check passed, and no conflict markers remain. Simulator code and dependencies did not change, so the earlier 11 passed simulator checks remain applicable. The parent independently reviewed the original #5 diff without a material finding. Refreshed independent integration review and installed-package verification remain parent integration work. The commit containing this refresh record changes documentation only.

## Progression slice #7

Ticket [#7](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/7) was implemented from verified lifecycle base `de46dd110f6a94229df4af5baa8ea619581138da` and refreshed onto verified catalog integration `497a426f50a16bc0b239f9fa843e38b24691ef59`. The merged implementation tested at HEAD `6196ef6` on `feat/mcp-7-progression` preserves both catalogs and their acceptance records. Its complete source suite passed 38 tests in 182.92 seconds:

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests -q
```

Official SDK evidence includes two complete seed-zero lobbies, identical accepted statuses and ordered baseline/drain actions with and without extra status reads, early controlled elimination with placement 8, autonomous remaining-lobby completion, terminal budget, rejection of further actions and duplicate start, preserved close outcome, and fresh start. The concrete adapter additionally proves capacity 14 leaves combat unstarted at exhaustion, explicit end_turn remains available, frozen post-combat own records survive native removal, and a native winner-cleanup fixture retains placement 1 and its own snapshot. Public terminal health and level remain separate from the private own projection.

Injected internal steps, partial native writes, atomic audit replacement failure, late tool-result failure, failed close mutation and bounded-progress exhaustion preserve the original environment, policies, module bindings, RNG, audit bytes and accepted native bytes. Retried progression matches an uninterrupted real-simulator run. Shared pool/player/encoder/action-handler/step-function/combat-RNG aliases remain intact. An actual unavailable audit destination is rejected before native construction; logged status reads preserve environment/native-directory/RNG identities.

A final transport patch makes rejection-audit failure return `log_unavailable` instead of swallowing it. The affected source protocol check passed one test after the complete-suite run. The installed extension was then rebuilt and the same installed failure check passed one test. These two files, and this verification record, are the working diff after tested HEAD `6196ef6`; the commit containing this record owns that final patch. Source, tests, scripts and configuration have combined SHA-256 `0969675ab912a2ff4de876e309d50c7cff64680dd015df1c6956c97e4755a71a`.

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests/test_protocol.py -k audit_and_native -q
```

A fresh CPU-only, noneditable environment at `/tmp/tft-mcp-install-7` installed the unchanged simulator and extension. Both distributions record empty `dir_info`, and imports from `/tmp` resolve inside its site-packages. Python 3.14.7, MCP 1.30.0, NumPy 2.5.3, PettingZoo 1.27.0 and Gymnasium 1.4.0 match the source checks. No GPU or model-hosting distribution was installed. The final extension was reinstalled after merging both catalogs:

```sh
uv venv /tmp/tft-mcp-install-7
uv pip install --python /tmp/tft-mcp-install-7/bin/python . './mcp/server[dev]'
uv pip install --python /tmp/tft-mcp-install-7/bin/python --reinstall-package tft-mcp-server './mcp/server[dev]'
```

Setuptools generated root build and egg-info artifacts during installation. Only those generated untracked artifacts were moved outside the checkout before the final scope check. Simulator and root source, packaging, mandatory dependencies and CI remain unchanged.

From `/tmp`, the final installed absolute console launcher passed 15 combined lifecycle, full-lobby, champion/trait and item checks in 127.39 seconds. The late rejection-audit patch was reinstalled and its affected installed check passed separately in 2.27 seconds.

```sh
env -u APPIMAGE TFT_MCP_TEST_COMMAND=/tmp/tft-mcp-install-7/bin/tft-mcp /tmp/tft-mcp-install-7/bin/python -m pytest -c /tmp/tft-mcp-worktrees/issue-7/mcp/server/pyproject.toml /tmp/tft-mcp-worktrees/issue-7/mcp/server/tests/test_protocol.py /tmp/tft-mcp-worktrees/issue-7/mcp/server/tests/test_champion_catalog.py /tmp/tft-mcp-worktrees/issue-7/mcp/server/tests/test_item_protocol.py -q
env -u APPIMAGE TFT_MCP_TEST_COMMAND=/tmp/tft-mcp-install-7/bin/tft-mcp /tmp/tft-mcp-install-7/bin/python -m pytest -c /tmp/tft-mcp-worktrees/issue-7/mcp/server/pyproject.toml /tmp/tft-mcp-worktrees/issue-7/mcp/server/tests/test_protocol.py -k audit_and_native -q
```

Relevant unchanged simulator checks passed 11 tests after the integration refresh, with two existing PettingZoo observation warnings. Scope and whitespace checks passed against the combined catalog base, including the final working diff:

```sh
env -u APPIMAGE PYTHONHASHSEED=0 /tmp/tft-mcp-env/bin/python -m pytest UnitTests/rng_test.py UnitTests/default_agent_test.py UnitTests/game_round_test.py UnitTests/simulator_test.py -q
env -u APPIMAGE /tmp/tft-mcp-env/bin/python mcp/server/scripts/check_scope.py 497a426f50a16bc0b239f9fa843e38b24691ef59
git diff --check
```

Initial tracer failure for missing planning budget was resolved. Two obsolete lifecycle assertions were reconciled with atomic buffering and audit preflight: failed candidate startup records remain unpublished, and diagnostic directories are identified by error details. One installed check invocation referenced nonexistent catalog filenames and ran no tests; the corrected combined command above passed. No unresolved executed check failed.

This bounded slice does not implement later inspection/action tools, runtime set selection, simulator schedule repairs, restart recovery, cross-file crash persistence, GitHub CI or required real Codex client acceptance. The owner subsequently rejected fixed set hardcoding; the parent owns scope reconciliation and readiness of affected contracts. Scheduling and recovery use the installed simulator and existing baseline without adding a set-specific rules table. Owner acceptance, PR integration and deployment are separate from the technical evidence recorded here.

## Installed-simulator clarification after #7

The owner clarified on 2026-10-09 that the extension uses the installed simulator without a fixed set number. Current tool descriptions, package/module descriptions, shared and server guides, setup prompt and Spec now follow that direction. No runtime set selector, guard, copied catalog, core change or generic multiset framework was added. Existing concrete mechanics remain supported through their installed definitions. Kayn form metadata now excludes form-item IDs absent from `item_stats.items`.

Simulator identity records `environment_name` directly from `TFT_Simulator.metadata`. The tested installed source currently reports `tft-set4-v0`; this is observed source metadata, not a server set constant or compatibility guard. Source digest and distribution/revision evidence remain recorded.

The final clarification diff was tested against precommit HEAD `c866647`, with combined integration base `497a426f50a16bc0b239f9fa843e38b24691ef59`. Server source, tests, scripts and configuration have combined SHA-256 `2997a48fc27245073e55129dafa5193703e67240762a2582bf53572815b38d4b`. The commit containing this clarification record owns the tested diff.

Affected catalog and lifecycle source checks passed 20 tests. The final noneditable extension was reinstalled, and the same checks through the installed launcher from `/tmp` passed 20 tests in 21.06 seconds. These checks include generic MCP descriptions, absence/rejection of a set selector, actual environment metadata, omission of absent special items and existing exhaustive source-definition comparisons:

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests/test_champion_catalog.py mcp/server/tests/test_item_catalog.py mcp/server/tests/test_item_protocol.py mcp/server/tests/test_protocol.py -k 'not seed_zero_full_lobby' -q
env -u APPIMAGE TFT_MCP_TEST_COMMAND=/tmp/tft-mcp-install-7/bin/tft-mcp /tmp/tft-mcp-install-7/bin/python -m pytest -c /tmp/tft-mcp-worktrees/issue-7/mcp/server/pyproject.toml /tmp/tft-mcp-worktrees/issue-7/mcp/server/tests/test_champion_catalog.py /tmp/tft-mcp-worktrees/issue-7/mcp/server/tests/test_item_catalog.py /tmp/tft-mcp-worktrees/issue-7/mcp/server/tests/test_item_protocol.py /tmp/tft-mcp-worktrees/issue-7/mcp/server/tests/test_protocol.py -k 'not seed_zero_full_lobby' -q
```

Focused real-simulator lifecycle/progression checks passed another 14 tests after the metadata change. Complete seed-zero lobbies were not rerun for the later labeling/metadata clarification because scheduling and recovery were unchanged; their preceding complete source and installed evidence remains recorded above.

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests/test_session.py mcp/server/tests/test_progression.py -k 'not real_seed_zero' -q
```

An independent focused source/test review at checkpoint `11e60e1` found no material atomicity issue and confirmed audit/native preflight, non-failing post-replacement publication, full aggregate alias recovery and final snapshot completeness. The parent retained its report at `/tmp/tft-mcp-context/review-7-atomicity.md`. That review is distinct from final parent review of the clarified contract and whole-milestone acceptance.
