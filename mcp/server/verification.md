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

## Own inspection slice #3

Verified on 2026-10-09 in `/tmp/tft-mcp-worktrees/issue-3`, branch `feat/mcp-3-inspection`, against fixed prerequisite base `7215c6a709d0075ed56408a8656865724aafe318`. The worktree was clean on that exact base before writing. Reviewed inspection contracts were recorded in the Spec before implementation. The accepted seams are the production-launcher MCP SDK and focused concrete GameSession fixtures. TDD added one category at a time with an observed failing tracer before its implementation, then added a failing nonempty shop-consistency test before its guard.

Prerequisite evidence supplied by the parent: PR #17 integrated #7 as `7215c6a709d0075ed56408a8656865724aafe318`. Exact slice `abed92714d629faf9f80c1e1fe20d328a04fb066` passed all 40 server tests, including complete lobbies. The merger confirmed identical Git tree `bb5193f87adc0e2d3bca5354e2fb466e933c9ae0` and passed 38 additional integrated checks with only two unaffected full-lobby repeats deselected. Scope covered 14 MCP paths and whitespace passed. Those prerequisite checks were not rerun wholesale for inspection.

Implementation and final affected tests are committed at `d2cf5a93796e369212a0aba94431ea7d0df92a7d`. The focused suite ran on the identical working tree subsequently committed at that head. After committing, merging local `feat/mcp-server-main` reported already up to date at the fixed prerequisite base. The exact committed head passed combined quick regressions. Parent reviewed the session, transport, Spec and both new test files and found no material issue. The evidence-only commit containing this record preserves that tested implementation and test tree. Server Python/configuration files have SHA-256 `7ee8f30a97f5520340155954c25fe84d5e550878d7c83c3ca3add7ccaa4d2ade`, computed in sorted relative path order, hashing each path followed by bytes and excluding `__pycache__`.

The final focused suite passed 17 tests in 63.95 seconds. It covers all seven separate strict tools, native seeded offers/prices and economy/trait values, explicit empty slots, board coordinates, Chosen/items/four-star/Kayn/sandguard allowlists, output schema validation, exhausted shared budget, exact retained terminal records, nested mutation isolation, negative native terminal health, close/restart, actionable rejection errors and audit-failure retry. Pickled complete game state remains byte-identical across repeated adapter reads and rejected selectors. Python/NumPy RNG, episode RNG, baseline RNG, module bindings, caches and graph aliases remain unchanged. Two following real rounds match a no-read reference game.

The actual production-launcher SDK seed-zero terminal run retained own category round 13, health 0 and null budget, while get_round and lifecycle status reported final lobby round 30. Repeated category calls returned identical data until close. The separate native negative-health fixture proves retained health is not clamped by the adapter. These are observed fixture values, not new gameplay constraints.

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests/test_inspection.py mcp/server/tests/test_inspection_protocol.py -q
```

Exact-head combined checks passed 54 tests with three repeats deselected in 36.81 seconds. The two existing complete-lobby checks retain prerequisite evidence; the new terminal inspection check has its own final full-run evidence above.

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests -k 'not seed_zero_full_lobby and not real_seed_zero and not terminal_inspection' -q
```

Relevant unchanged simulator checks passed 11 tests in 2.13 seconds, with two existing PettingZoo observation warnings. The six-path implementation scope and whitespace checks passed against the fixed base; final evidence adds only this MCP verification record.

```sh
env -u APPIMAGE PYTHONHASHSEED=0 /tmp/tft-mcp-env/bin/python -m pytest UnitTests/rng_test.py UnitTests/default_agent_test.py UnitTests/game_round_test.py UnitTests/simulator_test.py -q
env -u APPIMAGE /tmp/tft-mcp-env/bin/python mcp/server/scripts/check_scope.py 7215c6a709d0075ed56408a8656865724aafe318
git diff --check 7215c6a709d0075ed56408a8656865724aafe318
```

Initial missing-tool tracer failures and the shop-mismatch failure were resolved. Early test fixtures incorrectly used the champion constructor `items` keyword and a nonexistent `target_dummy` champion name; final fixtures use the native constructor and `sandguard` with its target-dummy flag. A baseline RNG assertion initially compared NumPy arrays directly and now compares serialized states. The first terminal test incorrectly required strictly negative health; the observed zero-health elimination is allowed, and the corrected full run passed. No executed check remains failed.

These checks use Python 3.14 with the installed unchanged editable simulator and CPU dependencies in `/tmp/tft-mcp-env`. Fresh SDK subprocesses run the actual launcher from a temporary working directory with this checkout's source path; APPIMAGE is removed for every command. The extension was not reinstalled into a new standalone environment for #3, so this record does not claim fresh noneditable packaging acceptance. No Codex or Claude Code interactive game, owner gameplay acceptance, PR integration, deployment or GitHub CI was executed. Those remain separate milestone evidence. No simulator/core, root packaging or shared environment package changes occurred.


## Verified pause after #3

The owner requested a graceful stop after #3 on 2026-10-09. PR #18 merged at `684bbbf8e88be96890b0069c0bc7d4728ff0ceff`. The independent merger repeated 54 combined quick tests with exactly three documented full-lobby repetitions deselected, confirmed identical slice/merge tree `e206d95f2c2959b7bda37fd01c340b4f92e97938`, then passed 16 integrated inspection checks with the previously tested terminal repetition deselected. Fixed-base scope and whitespace passed for seven allowed paths. No material review finding remained in the slice. The complete affected 17-test terminal evidence and 11 simulator checks above remain applicable.

No dependent implementation, final whole-milestone code-review, all-family replay or actual Codex/Claude LLM gameplay started. [CONTINUATION.md](CONTINUATION.md) records the verified frontier, environment, design contracts and remaining work. The pause commit changes documentation only; runtime checks were not repeated for that prose-only change. Scope, whitespace, tracker consistency and clean-worktree cleanup are checked for the saved pause.

## #4 public player inspection

Reviewed base `bde19766296fe70b7ead45cd57057da7bc0dfd61`, functional head `7384dcb04a60ff733619df30e3634d590bbea37d`, branch `feat/mcp-4-public-inspection`, isolated worktree `/tmp/tft-mcp-worktrees/issue-4`. The verified initial worktree was clean. A final merge of `feat/mcp-server-main` reported already up to date at the same base. All seven functional paths are under `mcp/`, and simulator source, root packaging, dependencies and CI remain unchanged. The functional diff SHA256 from `git diff bde19766296fe70b7ead45cd57057da7bc0dfd61 7384dcb04a60ff733619df30e3634d590bbea37d | sha256sum` is `aaeb3f63f614e23ebae6526204e66758ef1a7abf9d826a8b61f9033291a78f7a`. Functional commit working diff was empty; this record is the only subsequent documentation change.

Transport registers strict no-argument `get_players` and extends existing board/trait selectors. Concrete session methods resolve sorted initial dictionary keys and read only public opponent fields under the existing lock. Removed records use #7 native placements and public final scalars; own terminal categories retain their existing snapshot. The Spec contract was recorded before code, and current own-only descriptions and assertions were reconciled.

The accepted production SDK and focused concrete adapter seams were used with vertical TDD. The first player-list test failed on missing discovery, then passed after registration and projection. The opponent SDK test failed on `invalid_player`, then passed after public resolution. Fixture failures were corrected without product changes: subprocess working directories needed creation, early baseline boards remained empty until later rounds, a native Chosen constructor produced two stars, and SessionError text is read through `str(error)`. No executed check remains unresolved.

The environment is the retained CPU-only Python 3.14.7, MCP 1.30.0, NumPy 2.5.3, PettingZoo 1.27.0, Gymnasium 1.4.0 and pytest 9.1.1 environment. Every Python command removes inherited APPIMAGE. Source checks select the slice extension through PYTHONPATH without reinstalling shared packages.

The new SDK suite passed all four tests in 64.57 seconds. It proved discovery, strict inputs, all stable public IDs, local living-opponent board coordinates, visible nonempty boards, stored traits, unchanged later progression across two real server processes, audit failure/retry, actual terminal removals and a removed winner, own retained categories, close and fresh-start cleanup. Focused adapter tests passed all three in 1.16 seconds before the final recursive-schema assertions. These cover distinctive hidden fixture values, nested schema rejection and defensive copies, dictionary-key identity, complete graph/RNG/cache/module-binding purity, and native forced removals and winner records.

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests/test_public_inspection_protocol.py -q
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests/test_public_inspection.py -q
```

Final functional-source combined checks passed 60 tests with four complete-lobby repetitions deselected in 47.57 seconds. This includes the final recursive schema assertions and status/error wording. Prerequisite complete-game evidence remains applicable; the affected terminal public scenario receives its own exact-head refresh below.

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests -k 'not seed_zero_full_lobby and not real_seed_zero and not terminal_inspection and not terminal_public_list' -q
```

The selected unchanged simulator checks passed 11 tests in 2.33 seconds with the same two existing PettingZoo observation warnings. Existing broader Gymnasium failures are unchanged and were not rerun for this slice.

```sh
env -u APPIMAGE PYTHONHASHSEED=0 /tmp/tft-mcp-env/bin/python -m pytest UnitTests/rng_test.py UnitTests/default_agent_test.py UnitTests/game_round_test.py UnitTests/simulator_test.py -q
```

Fresh noneditable installation, actual CLI LLM gameplay, all action families, full-milestone replay and final milestone review remain unexecuted and assigned to their later accepted slices. #4 establishes public inspection only. Parent Standards/Spec review, PR creation and separate integration remain pending at writer handoff; no push, PR or issue mutation was performed by this writer.

The exact functional head terminal refresh passed one test with three deselected in 54.92 seconds. The actual seed-zero lobby completed at round 30 with controlled placement 8 and winner `player_2`, health 5, level 8, placement 1. All eight stable participant records retained their native placements; removed winner board/traits returned `player_eliminated`, and own frozen categories remained available. These are observed values, not fixed test assumptions.

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests/test_public_inspection_protocol.py -k terminal_public_list -q
```

Final scope passed for eight changed paths including this record, and whitespace passed against the fixed base. After the prose-only verification commit, source/tests/config are unchanged from the reviewed functional head and the working diff is empty.

```sh
env -u APPIMAGE /tmp/tft-mcp-env/bin/python mcp/server/scripts/check_scope.py bde19766296fe70b7ead45cd57057da7bc0dfd61
git diff --check bde19766296fe70b7ead45cd57057da7bc0dfd61
```


### Standards review for #4

Zero material findings. The independent reviewer inspected all seven functional paths, the subsequent verification prose, repository standards and the full smell baseline. The diff follows mcp/AGENTS.md and the Spec's concrete Python and ownership contracts. Transport owns strict schemas and delegation; GameSession owns public projection and locking. Shared unit/own projections are reused. No framework, generic dispatcher, copied catalog, dependency reversal or outside-scope source/configuration was introduced. The documented adapter responsibility supersedes Feature Envy suggestions to move code into core. Existing selector branches and transport dispatch do not justify unrequested abstraction. No runtime checks were run by this Standards reviewer.

### Spec review for #4

Zero material findings. The independent reviewer found no missing accepted requirement, scope creep or incorrect behavior. Sorted initial dictionary keys, explicit public scalar/unit allowlists, existing public_final/placements, preserved own terminal data, local coordinates and removed-opponent errors match #4 and the #3/#7 contracts. Strict transport schemas and focused SDK/native fixtures preserve privacy and read purity. The reviewer independently passed all three focused real-adapter public inspection tests in 1.16 seconds. Recorded terminal and broader checks were inspected without repeating them. Fresh installed packaging, whole-milestone replay and actual LLM host acceptance remain later evidence.

Both reviews pinned functional head `7384dcb04a60ff733619df30e3634d590bbea37d` against base `bde19766296fe70b7ead45cd57057da7bc0dfd61`, using `git diff <base>...<functional-head>`. Reviewed artifact head `3e392a9ee921d72d838127c8f478fe67b0ad809c` changes only verification prose after the functional commit; source/tests/config are identical, and staged/unstaged/untracked status was clean. The commit containing this aggregate adds only this review record. Scope and whitespace are refreshed for that prose-only change, without new runtime checks. Standards: 0 findings, no worst issue. Spec: 0 findings, no worst issue. Full reports remain supplemental in /tmp/tft-mcp-context/review-4-standards.md and review-4-spec.md. This bounded slice review does not replace final milestone review.

## Verified integration and pause after #4

The owner resumed only the next ticket, #4, and requested another stop afterward. PR [#19](https://github.com/KyleDerZweite/TFTMuZeroAgent/pull/19) merged only into the fork's `feat/mcp-server-main` at `f4a7de26e88b80468aa02ea45e1ad8c63fbdb617` on 2026-10-09. The independent merger verified clean root and writer worktrees, the fork, exact PR head/base, latest integration ancestry, contracts, all eight changed paths and both review axes. The merge used the exact checked head `22016669f3612ec5ead86a631a39b147e67204b6`. Integration and slice trees are identical at `3453db57846b0ed32b9699192eb20f5c6630b0c1`; root fast-forwarded to the merge.

Before merge, the exact slice head passed 60 combined quick checks with four documented full-lobby repetitions deselected in 42.10 seconds, using the command above. After integration, six public checks passed with the already-tested terminal repetition deselected in 9.44 seconds. These cover three real adapter and three actual production SDK scenarios. The functional-head full terminal journey and 11 simulator checks remain applicable because later changes are verification prose only. The merger found no material concern; independent Standards and Spec reviews each have zero findings.

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests/test_public_inspection.py mcp/server/tests/test_public_inspection_protocol.py -k 'not terminal_public_list' -q
```

Scope and whitespace passed before and after merge against `bde19766296fe70b7ead45cd57057da7bc0dfd61`. GitHub status checks were empty; CI was not executed under the accepted local-only verification choice. Fresh noneditable installation at the eventual final head, all-family replay, final whole-milestone review and actual Codex/Claude LLM gameplay remain unexecuted. Existing broader Gymnasium baseline failures remain documented above and were not rerun or repaired in this slice. No #8/#9 writer or dependent implementation started, and no main/upstream merge occurred.

The new pause commit changes only the Spec status, this evidence and [CONTINUATION.md](CONTINUATION.md). Runtime checks are not repeated for that prose-only checkpoint. Scope, whitespace, unchanged simulator/root packaging, tracker consistency, PR links and clean-worktree cleanup are checked before saving the pause. Ticket #4 is completed from the verified integrated results; milestone #1 remains open. The continuation record retains the next #8/#9 frontier and later acceptance requirements.

## Buy and sell slice #8

Verified locally on 2026-10-09 in isolated writer `/tmp/tft-mcp-worktrees/issue-8`, branch `feat/mcp-8-buy-sell`, against reviewed integration base `42cd697e70c79ed25dedc8bf67556fc60b858783`. Fork origin is `KyleDerZweite/TFTMuZeroAgent`; simulator sources, root packaging, dependencies and CI remain unchanged. Functional head is `4e96017713e0c2dbf0bfe871ee2bab747a4a7d75`. Its working diff was empty before this evidence-only record. `git merge feat/mcp-server-main` reported already up to date at the same integration base. No push, PR, issue mutation, upstream submission or integration merge was performed by this writer.

The accepted seams are the production MCP SDK stdio client and focused concrete GameSession fixtures. Tool discovery and strict recursive output schemas passed. SDK journeys cover a native affordable shop purchase and bench sale, malformed arguments and actionable rejection errors, audit failure with equivalent retry, and purchase followed by native board autofill/combat and board sale. Individual actions return in the same round, consume one planning slot, and preserve the controlled decision. The adapter fixtures also cover ordinary, cascading, Chosen and full-bench merges despite the actual disabled native buy mask; native pool/catalog/Chosen equivalence; promoted sale copy quantity and pool saturation; board/bench equipment insertion, bench whole-set overflow drops and glove normalization; Azir sandguard removals; strict direct requests; lifecycle/budget; unsupported dummies and price/promotion ranges; inconsistent native records; and deterministic actions with extra/reordered reads and rejection.

Both source-proven corrupt native merges reject with capacity_exceeded and preserve the committed graph, RNG and budget. Capacity tests cover a bench return filling the final inventory slot before board returns, a bench whole-set drop leaving room for board returns, and a higher cascade phase with board equipment. Copy postconditions reject corrupted owned quantities before commit. Buy and sale failure injection covers real native mutation before observation, after observation/mask mutation, after real baseline progression, copy postconditions, receipt construction, native filesystem writing and audit replacement. Each failure preserves committed aggregate identities, process/episode RNG, native/audit logs and module bindings, then retries equivalently to a fresh reference. Successful candidates retain simulator RNG, pool, player manager, observation and action-handler aliases.

Vertical TDD observed these failures before their fixes: ordinary adapter tracer lacked buy_unit; both corrupt merges originally returned success instead of capacity_exceeded; SDK discovery lacked both tools; a mismatched sale catalog count originally succeeded; and a nonempty shop string with null offer returned empty_slot rather than internal_error. Each gained the smallest corresponding implementation and passed before the next behavior. A temporary test called the nonexistent native buy_action instead of buy_shop_action; that fixture typo was corrected. Initial SDK setup incorrectly assumed seed-zero startup gold could buy an offer; the real journey now explicitly advances planning to earn gold. A rollback fixture initially captured module bindings before a later fixture setup; setup now precedes the snapshot. These transient checks failed and were corrected; they are not unresolved product failures.

All commands use the existing `/tmp/tft-mcp-env` without installing shared packages. Source/protocol tests launch the actual production launcher from a temporary working directory through the real MCP SDK. A fresh noneditable package installation and real Codex/Claude LLM gameplay remain later milestone evidence, not slice acceptance claims. Four unchanged whole-lobby repetitions are deliberately deselected. This slice adds a bounded real combat journey, not a new complete-lobby repetition. Independent Standards/Spec review and separate integration remain pending at writer handoff. GitHub CI was not executed under the accepted local-only verification choice.

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests/test_buy_sell.py mcp/server/tests/test_buy_sell_protocol.py -q
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests -k 'not seed_zero_full_lobby and not real_seed_zero and not terminal_inspection and not terminal_public_list' -q
env -u APPIMAGE /tmp/tft-mcp-env/bin/python mcp/server/scripts/check_scope.py 42cd697e70c79ed25dedc8bf67556fc60b858783
git diff --check 42cd697e70c79ed25dedc8bf67556fc60b858783
```

The final focused source/protocol suite passed 46 checks in 11.42 seconds. The final combined suite passed 106 checks with four unchanged repetitions deselected in 59.18 seconds on functional head `4e96017713e0c2dbf0bfe871ee2bab747a4a7d75`. Prior combined revisions passed 96, 102, 103 and 105 checks, with the same four unchanged repetitions deselected; those are intermediate evidence, not the final-head claim.

The unchanged simulator purchase, sale, bench, action-mask and action-dispatch suites passed 27 checks in 1.08 seconds. They ran from `/tmp/tft-mcp-context/issue-8-simulator-runtime` to contain native relative logs. The initial repository-cwd run passed the same 27 checks in 1.31 seconds but created an ignored writer-root log.txt; that generated artifact was removed and the isolated-cwd check repeated. No simulator/core repair was made.

```sh
cd /tmp/tft-mcp-context/issue-8-simulator-runtime
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=/tmp/tft-mcp-worktrees/issue-8/mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c /tmp/tft-mcp-worktrees/issue-8/mcp/server/pyproject.toml /tmp/tft-mcp-worktrees/issue-8/UnitTests/player_test.py /tmp/tft-mcp-worktrees/issue-8/UnitTests/bench_full_repro_test.py /tmp/tft-mcp-worktrees/issue-8/UnitTests/shop_buy_mask_test.py /tmp/tft-mcp-worktrees/issue-8/UnitTests/action_space_test.py /tmp/tft-mcp-worktrees/issue-8/UnitTests/step_function_test.py -q
```

The functional scope check passed five changed paths against the fixed base, including both new tests; whitespace passed. The final evidence-only commit adds this sixth server path. The complete scope check passed all six paths and whitespace passed after this record; source/tests/config are unchanged from the functional head. No #9/#10/#11 implementation was started. Native corrupt merge behavior remains in the unchanged simulator; the adapter rejects those unsafe requests atomically. Bench-sale equipment drops remain native supported behavior and are explicit in receipts.


### Standards review for #8

Zero material findings. The independent reviewer inspected every functional hunk and both complete new test files, repository standards and the full smell baseline. The Spec explicitly assigns simulator validation, projections and recovery to the concrete adapter, so native field access is accepted. Small price, catalog, inventory and receipt helpers share actual action responsibilities. Bounded buy/sell scaffolding does not justify a speculative common executor. Transport owns schemas, validation delegation and dispatch. No framework, new transaction, scheduler, generic dispatcher, outside-scope source or dependency change was introduced. README documents implemented calls in English; tests use real simulator fixtures and the production SDK boundary. No runtime checks were run by this Standards reviewer.

### Spec review for #8

Zero material findings. The independent reviewer found no missing material requirement, scope creep or incorrect implementation. Strict request/receipt shapes, native coordinates and pricing, merge-capacity ordering and Chosen promotion, full-bench mask disagreement, sale return/drop semantics, copy/inventory postconditions and the existing outer transaction match the accepted contract. Both corrupt merge fixtures reject before native execution. Coverage includes higher-phase capacity, ordinary/cascading/Chosen/full-bench merges, gloves, Azir, promoted pool quantities and saturation, lifecycle/budget, rollback with fresh-reference retry and read/rejection determinism. The reviewer independently passed all 46 focused real-adapter and production SDK checks in 11.54 seconds. Combined 106-check and simulator 27-check results remain writer evidence. Fresh final-head installation, whole-milestone replay, final milestone review and actual LLM-host gameplay remain later acceptance.

Both reviews pinned functional head `4e96017713e0c2dbf0bfe871ee2bab747a4a7d75` against fixed base `42cd697e70c79ed25dedc8bf67556fc60b858783` with `git diff <base>...<functional-head>`. Both inspected clean artifact `9ad89718ec674e87f27bfe40c8935bd3f7d6dd57`, whose subsequent delta changes only this verification record. Source/tests/config are identical and staged/unstaged/untracked changes were empty. The commit containing this aggregate adds only review prose; scope and whitespace are refreshed without repeating runtime checks. Standards: 0 findings, no worst issue. Spec: 0 findings, no worst issue. Full supplemental reports are /tmp/tft-mcp-context/review-8-standards.md and review-8-spec.md. This bounded review does not replace final milestone review.

## Verified integration and pause after #8

PR [#20](https://github.com/KyleDerZweite/TFTMuZeroAgent/pull/20) merged only into the fork's `feat/mcp-server-main` at `a208efdc89353f248d0d92d358e15453089a4467` on 2026-10-09. Its exact parents are reviewed base `42cd697e70c79ed25dedc8bf67556fc60b858783` and checked slice `8048bddb18024a88b3d016bbba00146d8a7df2d8`. The independent merger inspected all six changed paths, native mechanics, accepted contracts and both review reports. Before the exact-head guarded merge, it verified the fork, clean workspaces and latest integration ancestry, then independently passed 106 combined checks with four unchanged whole-lobby repetitions deselected in 59.27 seconds.

The complete slice and merge Git trees match at `ed7493e0c6469f7367bd32d80db97563fa011a04`. Root fast-forwarded cleanly, then all 46 focused adapter and production SDK checks passed on integrated root in 11.48 seconds. Scope and whitespace passed before and after merge for six server paths. No material concern remained. The 27 unchanged simulator checks and both independent review reports remain applicable; later slice commits change verification prose only. Source/tests/config match pinned functional head `4e96017713e0c2dbf0bfe871ee2bab747a4a7d75`.

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests/test_buy_sell.py mcp/server/tests/test_buy_sell_protocol.py -q
```

GitHub status checks are absent under the accepted local-only workflow. Four unchanged completed-lobby repetitions, fresh final-head noneditable installation, all-family replay, final milestone review and actual Codex/Claude LLM gameplay were not executed for this slice. Earlier broader Gymnasium baseline failures remain documented and were not rerun or repaired. Native corrupt merge behavior remains in the unchanged simulator, with unsafe requests rejected by the adapter. Native bench-sale equipment loss is supported and reported explicitly.

The owner requested only the next ticket after #4 and another stop afterward. Only #8 was implemented and integrated. Ticket #8 is completed from actual verified results; milestone #1 stays open. #9 remains the next reviewed ticket, while #10/#11 have their prerequisites integrated and still need detailed contract reviews before Ready. No subsequent writer started, no main/upstream merge occurred, and [CONTINUATION.md](CONTINUATION.md) retains the remaining contracts and evidence requirements.

The pause checkpoint changes only Spec status, this evidence and the continuation record. Runtime checks are not repeated for those prose-only changes. Before saving the pause, the parent checks fixed-base scope and whitespace, unchanged simulator/root packaging, ticket and milestone consistency, PR links, clean ancestor-preserving worktree removal and pushed clean integration.

## Shop refresh and experience slice #9

Implemented on 2026-10-09 in isolated writer `/tmp/tft-mcp-worktrees/issue-9`, branch `feat/mcp-9-shop-xp`, clean reviewed base `c9ff68c5b5d8778393ff8c0693093d4f481f7305`. Functional head is `c98cb7cd761924b84177bbcd6894d97e31839cc0`. The integration branch still pointed to the base when the writer merged its latest tip and received `Already up to date`. This record changes evidence prose only; source, tests and configuration are frozen for independent review.

`refresh_shop` and `buy_xp` use strict no-argument MCP schemas, concrete session methods, the existing outer transaction and one nested controlled action. The adapter reads native instance costs, cap and thresholds. Refresh verifies both shop lists regenerate and projects the five actual offers through existing shop validation. XP verifies observed crossed-level conservation and capacity increments without copying native leveling. Cap errors precede affordability after lifecycle and budget checks. No simulator, root packaging, dependency, CI, scheduler or RNG owner changed.

Vertical TDD reds were observed before implementation: refresh lacked a GameSession method, four XP fixture cases lacked a GameSession method, and production SDK discovery lacked refresh registration. After registration, missing manual no-argument dispatch validation returned invalid_input instead of no_game; adding the tools to existing validation fixed it. The first earned-gold fixture used two turns and correctly received insufficient_gold after refreshing. Three native turns provide the intended successful refresh and level-transition journey. Initial error-detail assertions omitted the existing transaction diagnostic paths; assertions now select the contractual resource/cap fields. These fixture corrections changed no simulator behavior.

The focused suite passed 34 checks in 13.30 seconds before a test-only fault label was clarified. It includes production stdio discovery, strict requests, earned-gold refresh/XP, native level transition, audit failure/recovery and fresh-process refresh replay despite reordered reads and rejections. A separately named SDK memory-stream fixture runs unchanged transport.serve with a preconfigured real session for near-cap success and atomic cap rejection. That fixture proves protocol behavior, not naturally reached production stdio cap gameplay.

Real adapter fixtures cover nonleveling and recursive multilevel XP, discarded cap overflow, bonus capacity, changed native costs and thresholds, affordability/cap/budget/lifecycle precedence, silent native no-ops and corrupt gold/shop/XP/level/capacity effects. Eight fault modes per action cover native mutation, observation updates, real baseline progression, postcondition inspection, native writes and audit commit. Snapshots include native episode NumPy generator state in addition to existing gameplay, episode Python and baseline RNG, process Python/NumPy state, graph aliases, module bindings, budget and accepted log bytes. Recovery retries match fresh real reference sessions.

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests/test_shop_xp.py -q
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests -k 'not seed_zero_full_lobby and not real_seed_zero and not terminal_inspection and not terminal_public_list' -q
```

An earlier combined snapshot passed 139 checks with four unchanged completed-lobby repetitions deselected in 63.79 seconds. It preceded the final production recovery test and explicit NumPy snapshot coverage. The exact frozen-head combined result is recorded below.

The unchanged native RNG, Player, step-function and action-space checks passed 27 tests in 0.86 seconds from external cwd `/tmp/tft-mcp-context/issue-9-simulator-runtime`. The first attempt could not start because that external directory did not exist; creating it and rerunning succeeded. No native check ran from the checkout or modified root log.txt.

```sh
cd /tmp/tft-mcp-context/issue-9-simulator-runtime
env -u APPIMAGE PYTHONHASHSEED=0 /tmp/tft-mcp-env/bin/python -m pytest -c /tmp/tft-mcp-worktrees/issue-9/mcp/server/pyproject.toml /tmp/tft-mcp-worktrees/issue-9/UnitTests/rng_test.py /tmp/tft-mcp-worktrees/issue-9/UnitTests/player_test.py /tmp/tft-mcp-worktrees/issue-9/UnitTests/step_function_test.py /tmp/tft-mcp-worktrees/issue-9/UnitTests/action_space_test.py -q
```

Scope passed five functional changed paths against the reviewed base, including the untracked test before commit. Whitespace passed. `git diff 33c2c6e -- Simulator UnitTests pyproject.toml` was empty. Final evidence-only scope includes this sixth path. No push, PR/issue mutation, integration merge, main/upstream change or later-ticket implementation was performed by the writer. Independent Standards/Spec review, integration and owner acceptance remain separate. Four unchanged whole-lobby repetitions, final noneditable installation, all-family replay and real LLM-host gameplay remain unexecuted for this bounded slice and belong to #12/#13. Existing broader Gymnasium baseline failures remain unchanged.

At exact frozen functional head `c98cb7cd761924b84177bbcd6894d97e31839cc0`, the combined command above passed 140 checks with the same four unchanged whole-lobby repetitions deselected in 67.59 seconds. This includes all 34 focused #9 scenarios after the fault-label clarification and native episode NumPy snapshot additions. Final six-path scope and whitespace passed with only verification prose changed since that tested head.

Independent Standards review checked all five functional changed files at `c98cb7cd761924b84177bbcd6894d97e31839cc0` against `c9ff68c5b5d8778393ff8c0693093d4f481f7305` in its own clean detached worktree. It found zero documented violations or actionable baseline smells and ran no runtime tests.

Independent Spec review used a separate clean detached worktree at the same head/base and found zero material findings. It independently passed all 34 focused #9 checks in 13.58 seconds. Production stdio evidence and the distinct real-session SDK cap fixture match the accepted seams. The parent verified final writer head `6c22b0141f83b94b0d179a0944659aec7bb61823` adds only this verification document relative to the reviewed functional head. The following aggregate commit adds review prose only; no source, test or configuration changes require refreshed runtime evidence. The parent had already closed an early coverage gap by requiring explicit native episode NumPy generator state in the focused rollback/rejection/retry fixtures before the functional freeze.


## Verified #9 integration and pause

PR [#21](https://github.com/KyleDerZweite/TFTMuZeroAgent/pull/21) merged only into fork feat/mcp-server-main at `c42a6fc273a1bb426d0844fd90da542120fdd173`. Native merge parents are exactly reviewed base `c9ff68c5b5d8778393ff8c0693093d4f481f7305` and final slice head `3f6aca1c05e47377e1b0ce2d4ab733f5c98ba40f`. Full merge and tested slice trees are identical at `2bfe212ac48ebb7697da34433baee6103fbf98ea`.

The independent merger reran the combined command at the exact PR head: 140 passed, four unchanged whole-lobby repetitions deselected, 69.23 seconds. After clean fast-forward, integrated root independently passed all 34 focused checks in 13.50 seconds using the focused command with `--basetemp=/tmp/tft-merge9-root-tests`. Six-path scope, whitespace, native merge parents, tree identity, unchanged Simulator/UnitTests/root packaging, remote tip and clean status passed. GitHub check runs and commit statuses are absent under the accepted local-only policy. No merger check failed.

Only #9 was implemented in this resumed run. Ticket #9 is closed from actual integrated evidence; milestone #1 remains open. #10/#11 need detailed contract review before Ready. No subsequent implementation started. Naturally reached production stdio cap gameplay, final noneditable installation, all-family replay and actual LLM-host gameplay remain unexecuted. Four unchanged completed-lobby repetitions were not repeated; existing broader Gymnasium baseline failures remain unchanged.

The parent saves this documentation-only pause after checking fixed-base scope and whitespace, unchanged simulator/root packaging, tracker and exact milestone mirror, PR links, clean ancestor-preserving removal of writer/review worktrees and pushed clean integration. Source/tests/configuration remain identical to the verified merge, so runtime evidence is retained. [CONTINUATION.md](CONTINUATION.md) owns the next slice and remaining acceptance.

## Movement slice #10

Implemented on 2026-10-09 in isolated writer `/tmp/tft-mcp-worktrees/issue-10`, branch `feat/mcp-10-movement`, from clean reviewed integration base `77a33afaa767f6122ac83c35f71f216b747a4dd3`. Functional head is `6146834f46438b297c53865bff8f629b4755c85d`. The writer verified its actual repository, branch, base and clean state before editing. Root integration was clean at the same base, and merging its latest tip reported `Already up to date`. Source, tests and configuration are frozen at the functional head; this record changes evidence prose only.

The accepted seams remain production MCP SDK stdio and focused real GameSession fixtures. Strict source/target locations and receipt schemas reuse existing location, Unit change and status definitions. Shared location validation preserves sell_unit. Movement uses the existing outer transaction, one nested controlled_action and native `[5,source_flat,target_flat]`. Legality checks and identity postconditions preserve native direction, first-vacancy displacement, full-bench swaps, regular-unit capacity, dummy restrictions, Azir guard/linkage behavior and glove tracking. No simulator rules, scheduler, budget, transaction or RNG owner was copied or replaced. Successful baselines can change the shared pool; success does not impose whole-pool equality.

Production stdio journeys acquire ordinary units through native purchases. They cover board orientation, empty board and bench targets, board swaps in both coordinate directions, occupied bench/board swaps in both requested directions, empty-source/no-op rejection, strict nested schemas and audit failure/recovery. A seeded nine-move tape replays in independent processes, comparing receipts, category checkpoints and actual baseline progression. Extra/reordered reads and representative no-op rejections preserve the same tape. Separate real-session fixtures cover full bench/capacity, earlier-vacancy displacement and directed rejection, visibly identical distinct-unit swaps with an empty delta, five glove-swap patterns, board dummies, Azir entry/removal/reposition and moving or swapping linked guards. One rare Azir fixture crosses official SDK memory streams through unchanged transport; it is not naturally acquired stdio Azir gameplay. Native Azir bench movement succeeds despite its actual disabled movement mask. Ordinary family fixtures compare real native Player outcomes, traits, full observation arrays and action masks.

Ten injected failure paths cover native mutation, observation and mask updates, baseline progression, silent native no-op, corrupt effects, receipt/postcondition inspection, native-file writing and audit replacement. Every failure preserves byte-identical accepted complete game graphs, including pools, encoders, traits and masks; accepted aggregate references and aliases; process/episode Python and NumPy RNG, baseline RNG and module bindings; planning budget; and native/audit log bytes. Fresh-reference retries compare gameplay and the complete native graph. Only native Player/champion wall-clock start_time and the exact Player.print clock field are normalized in that comparison. Player messages, CombatContext.log, combat milliseconds, ordering and gameplay values remain intact. Normalization edits cloned Player log lists in place to preserve aliases. A separate real fresh-session diagnostic confirmed combat-log and Player-message changes remain comparison-sensitive. Failure rollback comparisons use no normalization.

Observed TDD reds preceded the basic method, legality and protocol registration: one missing-method failure, four restriction failures and missing SDK discovery. They passed after the respective implementation. A test initially used await inside a generator consumed by next, producing an async-generator TypeError; gold is now read before selecting an offer. A native encoder comparison initially omitted the actual StepFunction action-counter decrement; the reference now includes it. Fresh-reference raw pickle bytes initially differed only on native wall clocks; those fields now have narrow type-specific normalization. An exploratory cloned-graph diagnostic differed on unordered-set pickle iteration; the proof was repeated with independently seeded real reference sessions, matching the accepted retry seam. The initial native command could not start because its external cwd was absent; creating that directory and rerunning succeeded. These corrected fixture/diagnostic failures are not unresolved product failures.

The parent identified two early draft findings before functional freeze. Spawned guard coordinate validation could return invalid_input for corrupt native results; direct postcondition validation now returns internal_error, and all four corrupt-Azir rollback cases passed. An unused spawned-guard list was removed. Generic log normalization could erase combat data; normalization now recognizes only native Player log clocks and Player/champion wall-clock fields. All ten rollback/fresh-retry cases passed after that correction, and the focused suite passed all 54 checks in 17.96 seconds. An earlier combined run passed 194 checks with four unchanged whole-lobby repetitions deselected in 82.62 seconds, at identical final source with the earlier normalization helper. Exact functional-head combined results are recorded below.

Verification uses the existing CPU `/tmp/tft-mcp-env`, Python 3.14.7, MCP 1.30.0, NumPy 2.5.3, PettingZoo 1.27.0, Gymnasium 1.4.0 and pytest 9.1.1. No shared packages were reinstalled. Extension source is selected through PYTHONPATH. SDK subprocesses run the production launcher outside the checkout. External native imports resolve to `/tmp/tft-mcp-env/lib64/python3.14/site-packages/Simulator`; all 53 installed Python files match the unchanged checkout. Fresh final-head noneditable extension installation remains #13 evidence.

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests/test_movement.py -q
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests -k 'not seed_zero_full_lobby and not real_seed_zero and not terminal_inspection and not terminal_public_list' -q
env -u APPIMAGE /tmp/tft-mcp-env/bin/python mcp/server/scripts/check_scope.py 77a33afaa767f6122ac83c35f71f216b747a4dd3
git diff --check 77a33afaa767f6122ac83c35f71f216b747a4dd3
git diff --exit-code 33c2c6e -- Simulator UnitTests pyproject.toml setup.py setup.cfg requirements.txt .github
```

Unchanged native RNG, Player, movement/step, action-mask/action-space and bench checks passed 29 tests in 1.13 seconds from external cwd `/tmp/tft-mcp-context/issue-10-simulator-runtime`, containing native relative logs outside the writer root.

```sh
cd /tmp/tft-mcp-context/issue-10-simulator-runtime
env -u APPIMAGE PYTHONHASHSEED=0 /tmp/tft-mcp-env/bin/python -m pytest -c /tmp/tft-mcp-worktrees/issue-10/pyproject.toml /tmp/tft-mcp-worktrees/issue-10/UnitTests/rng_test.py /tmp/tft-mcp-worktrees/issue-10/UnitTests/player_test.py /tmp/tft-mcp-worktrees/issue-10/UnitTests/step_function_test.py /tmp/tft-mcp-worktrees/issue-10/UnitTests/action_space_test.py /tmp/tft-mcp-worktrees/issue-10/UnitTests/bench_full_repro_test.py -q
```

Functional scope passed all five changed paths, including the untracked test before commit; whitespace passed. Simulator/UnitTests/root packaging and CI comparison against `33c2c6e` was empty. This evidence-only record adds the sixth path. Independent Standards/Spec review, integration and owner acceptance remain separate. Four unchanged complete-lobby repetitions, whole-milestone all-family replay, final installation and actual LLM-host games were not run for this bounded slice and remain #12/#13 work. Earlier broader Gymnasium baseline failures remain unchanged. No later slice, push, PR/issue mutation, integration merge, upstream change or worktree cleanup was performed by this writer.

At exact frozen functional head `6146834f46438b297c53865bff8f629b4755c85d`, the combined command passed 194 checks with the same four unchanged whole-lobby repetitions deselected in 79.49 seconds. This includes all 54 focused movement checks after the narrow native-clock normalization correction. Source/tests/config remained unchanged while only this verification record was written. Final six-path scope and whitespace passed, and the unchanged-core/root comparison remained empty.

The parent found a stale README introduction after the writer handoff, still describing positioning as a later slice. The introduction now lists move_unit and supported native positioning/swaps, with equipment remaining a later slice. This correction changes only README and this evidence record. Source/tests/config remain identical to functional head `6146834f46438b297c53865bff8f629b4755c85d`. Focused prose/link, scope and whitespace checks passed; runtime checks were not repeated for this prose-only correction.

Independent Standards review examined all five functional changed files at `6146834f46438b297c53865bff8f629b4755c85d` against reviewed base `77a33afaa767f6122ac83c35f71f216b747a4dd3` in its own clean detached worktree. It found zero documented violations or actionable baseline smells and ran no runtime tests.

Independent Spec review used a separate clean detached worktree at the same head/base, found zero material findings and independently passed all 54 movement checks in 17.48 seconds. The parent later found a stale guide introduction and the single writer corrected it at `c2a21fd407226ed58b42ad09e607edc92f9dcc8d`. The Spec reviewer checked that documentation delta and found zero remaining material findings. Relative to the reviewed functional head, the final artifact changes only server README and verification prose; source, tests and configuration are identical. This aggregate commit adds review prose only. Runtime evidence is retained, and the separate merger owns actual integration checks.


## Verified #10 integration and pause

PR [#22](https://github.com/KyleDerZweite/TFTMuZeroAgent/pull/22) merged only into fork feat/mcp-server-main at `93f1e071d2798956fb4bbb089bf61da04623281c`. Native merge parents are exactly reviewed base `77a33afaa767f6122ac83c35f71f216b747a4dd3` and final slice head `1ce146ee0e8f88121b212e15504844929f1c27f4`. Full merge and tested slice trees are identical at `67a59493d36fd1fea0723d007186747acf223ffd`.

The independent merger repeated the combined command on the exact PR head: 194 passed, four unchanged whole-lobby repetitions deselected, 52.05 seconds. After clean fast-forward, integrated root independently passed all 54 movement checks in 10.40 seconds with `--basetemp=/tmp/tft-merge10-root-tests-93f1e071`. Six-path scope, whitespace, unchanged native/UnitTests/root packaging/dependencies/CI comparison, merge parents/tree identity, clean status and remote/root tips passed. GitHub check runs and commit statuses are absent under the accepted local-only contract. No merger check failed.

The parent recreated the vanished CPU environment before implementation, installing the unchanged simulator from an external git archive of f4b194c. All 53 installed native Python files match the checkout. Baseline extension checks passed 140 with four unchanged whole-lobby repetitions deselected in 65.99 seconds. This establishes the recreated test environment, not final installed extension acceptance.

Only #10 was implemented in this resumed run. Ticket #10 is closed from actual integrated evidence; milestone #1 stays open. #11 requires its detailed contract review before Ready; #12 is still blocked by #11 and #13 by #12. No later implementation started. Naturally acquired production stdio Azir/glove scenarios, final noneditable extension installation, whole-milestone all-family replay and actual LLM-host games remain unexecuted for this bounded slice. The four unchanged completed-lobby repetitions were not rerun; existing broader Gymnasium baseline failures remain unchanged.

The documentation-only pause records the checked scope, whitespace, unchanged simulator/root, tracker and exact milestone mirror, PR links, clean ancestor-preserving removal of the writer and both review worktrees, and pushed clean integration. Source/tests/configuration remain identical to the verified merge. [CONTINUATION.md](CONTINUATION.md) owns the next slice and remaining acceptance.


## Resume and equipment contract review (#11)

The owner resumed all remaining tickets #11 through #13 on 2026-10-09 from clean integration `49e4a60d6bd9d44f88d6c9470ca75a8759f112e7`. Native GitHub edges remain #11:[6,8], #12:[4,5,9,10,11], #13:[12]. Both #11 blockers are closed and ancestor merges are retained. No dependent implementation has started.

A GPT-6.1-Sol medium reviewer used a clean detached worktree at this exact revision. Ten installed-native equipment probes and four trait-origin probes exited zero. They confirmed the accepted ordinary/special support and unsafe glove, recipe, trait-origin and Kayn cases. Safe native trait removal requires an intrinsic prefix and an equipment-grant suffix multiset. This reviewed legality refinement is recorded in the Equipment contract in the Spec; no core repair is needed. All 53 installed native Python files match the checkout. Relevant unchanged Player/RNG/action/step checks passed 27 tests in 0.66 seconds from an external cwd. These native probes establish contract feasibility, not equipment-tool acceptance. Supplemental probe reports are under `/tmp/tft-mcp-context/design-11-runtime/`.


### Equipment review refinements before integration

Independent initial Standards review found zero actionable points; initial Spec review at functional `a76d87fcaffe31d0377e6ee801f16152885c8913` found two P2 gaps, initial catalog validation and trait-changing postconditions. The writer reproduced both and added rollback/retry coverage. A parent SDK probe also showed an impossible stars-5 target could publish gameplay before output validation failed; preflight now retains four-star support and rejects unrepresentable stars. A native constructor probe independently confirmed process fallback field mutation, so preflight uses metadata and no preview constructor.

A further installed-native probe demonstrated an early board-contributor duplication cascade into three stars using discarded intermediate bench_loc=-1. The core produced an empty board with stale stored warlord/vanguard counts. This known unsafe case is explicitly unsupported in the revised Equipment contract; bench-only cascades and final-phase board contributors remain supported. No core, native placement or live cache repair is authorized or implemented. The contract refinement is consistent with #11's explicit audit/reject obligation. Source and acceptance review must refresh after these corrections before integration. Supplemental evidence is `/tmp/tft-mcp-context/constructor-context-probe.json` and `/tmp/tft-mcp-context/board-duplicate-cascade-probe/evidence.json`.
