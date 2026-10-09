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

## Equipment slice #11

Implementation started on 2026-10-09, with final checks on 2026-10-10, in `/tmp/tft-mcp-worktrees/issue-11`, branch `feat/mcp-11-equipment`, from clean reviewed base `8a47baf6e4c3113824e3e3aae40761f885d80bbd`. The writer verified the actual repository, branch, HEAD, empty working diff and fork origin before editing. Final functional head is `b0cdc8fc0aeb6c0fa6f7efe8d1c1e37f5ca0ee0b`, after source safeguard commits `a4413bffde6c82f3ec06b59621c471ea05cd7091` and `bfdfed5` and a merge of the reviewed contract checkpoint `da1a926784ff40d24bd374cd40ae7fb536982eeb`. Initial functional head was `a76d87fcaffe31d0377e6ee801f16152885c8913`. The writer merged the latest integration checkpoint `da1a926784ff40d24bd374cd40ae7fb536982eeb` before final verification. The original slice review base remains `8a47baf6e4c3113824e3e3aae40761f885d80bbd`; the final PR merge-base is the reviewed checkpoint. A task-local stash preserved unfinished verification prose during that merge. Its append conflict was resolved by retaining both the parent supplemental-audit section and this writer section, with no reset or discarded evidence. This record adds evidence prose only; source, tests and configuration remain identical to the functional head.

The accepted Equipment contract and native #6/#8 prerequisites govern this slice. The implementation uses strict item_slot/target validation, existing shared locations and schemas, the existing outer transaction, require_action_budget and one controlled_action `[6,target_flat,item_slot]`. Receipts contain exactly the accepted detached fields and all actual unit/inventory deltas. Native execution owns equipment effects, randomness, constructor/merge behavior, observations and action masks. No core mechanics, scheduler, transaction or RNG owner was replaced. All new implementation and tests are inside mcp/server; shared guide updates remain in mcp.

Production official SDK stdio scenarios naturally purchase Fiora and Garen in round 2, earn a sparring glove through combat, and equip ordinary board and bench units in round 3. They cover tool discovery, strict recursive schemas, lifecycle/input/empty-inventory rejection, exact receipts and budget, audit failure recovery, and seeded replay with extra reads and rejections. Replay compares both equipment receipts, all own-category checkpoints and every recorded baseline progression action. Separate test-local real-session official SDK memory fixtures cover native combination, remover, reforger, fresh duplicator, direct gloves and literal Kayn tokens. Those rare fixtures do not establish naturally acquired production consumable gameplay.

Focused real-session fixtures cover board/bench orientation, native last-component ordering, three-item recipes, force_of_nature without invented planning capacity, direct and recipe trait grants, safe removal in either equipment order, unsafe origin prefixes/suffixes, direct gloves and trackers, native reforger categories/exclusions/spatula, pre-consumption inventory capacity, default-star and Chosen duplication, safe cascading and board merges, native bench whole-set drops, Azir guard effects, all-board/all-token Kayn forms and repeat application, dummy restrictions, unknown or impossible records, four-star ordinary equipment, mask disagreement, lifecycle/budget errors and detached results. Actual native Player effects and its incremental observation/mask updates are compared in twelve board/bench variants, including traits and duplication.

Twenty fault cases cover native mutation, observation, mask and baseline updates, silent no-op, corrupt effects, receipt and postcondition inspection, native-file writes and audit replacement for random gloves and reforger. They preserve byte-identical full accepted graphs, aggregate identities and aliases, pools, traits, encoders/masks, policies/module bindings, process and episode RNG, budget and accepted native/audit log bytes. Fresh-reference retries compare complete gameplay and native graphs through the existing narrow Player/champion wall-clock normalization. Eleven additional corrupt-result cases reject false native success, equipment/origin/catalog/economy/shop/inventory/tracker/capacity/count changes and surviving-unit corruption during duplication. Process default CombatContext serialization and identity remain unchanged on accepted and rejected duplicator calls.

The parent found an early draft preview constructor outside simulator_scope. Its isolated installed-native probe at `/tmp/tft-mcp-context/constructor-context-probe.json` confirmed that construction writes `field_coordinates[-1][-1]` and binds the process fallback context. Preflight now creates only a SimpleNamespace with the minimal source-defined constructor metadata; it constructs no live champion, performs no RNG draw and initializes no combat field or queue. The actual constructor runs once through native controlled_action. Default-context regressions pass on success and rejection. The parent also identified inappropriate reuse of sale-price ranges for ordinary equipment. Target validation now reads champion definitions directly, and the real four-star equipment scenario passes. These findings were fixed before functional freeze.

Independent first-round Standards review reported zero findings. Spec review at `a76d87fcaffe31d0377e6ee801f16152885c8913` reported two P2 findings with actual native reproductions, recorded in `/tmp/tft-mcp-context/review-11-spec.md`. Native trait grant/removal/duplication could commit corrupted composition and tiers, and ordinary equipment accepted an initially inconsistent triple catalog. Seven focused regressions reproduced those missing rejection invariants before the fix. Every applicable equipment mode now validates owned-unit/catalog consistency before execution. Trait-changing operations validate native counts using `Player.team_origin_class` on a shallow player view with a fresh local composition dictionary, then calculate tiers from installed thresholds. A detached target item override preserves native grant timing before equipment insertion, including subsequent native double counting. No published cache, module binding, live champion or RNG is mutated by this inspection. Grant/removal/duplicate corruption cases preserve graph identities, RNG, logs and fresh-reference retry. All 115 focused equipment checks passed in 23.01 seconds after these corrections.

A supplemental parent native probe at `/tmp/tft-mcp-context/board-duplicate-cascade-probe/evidence.json` found unsafe early board-contributor cascades. Native outer transfer uses a discarded intermediate unit's bench_loc=-1 after a later promotion and can leave the board empty with stale counts or move an unrelated last bench unit. The existing merge preflight now records board participation per promotion phase and rejects any board contribution before the final phase with unsupported_action and reason early_board_duplicate_cascade. Bench-only cascades and final-phase board cascades remain supported. Explicit real-session rejection and safe-final-phase scenarios pass. An added native-record check rejects boolean catalog counts/levels that Python otherwise compares equal to integers. The accepted Spec, catalog constraints and guide record the cascade limit. All 120 focused equipment checks passed in 23.25 seconds before the documentation-only contract merge. Intermediate head `a4413bffde6c82f3ec06b59621c471ea05cd7091` passed 309 combined checks with four deselections in 118.56 seconds; it does not establish the later cascade safeguard.

The parent also found the output boundary for impossible native star values. An official SDK real-session stars-five fixture returned an unstructured SDK output-validation error after the game had already changed. The failing graph-equality regression reproduced that atomicity breach. Target validation now rejects values outside the accepted Unit schema's one-through-four range as internal_error before native execution. The SDK regression proves unchanged graph and status with the structured error, while four-star ordinary equipment remains supported. This protection is separate from sale-price limits.

TDD reds preceded each ordinary, recipe/trait, consumable, duplicator/Kayn and SDK registration capability. Corrected fixture failures included an incorrect assumption that two spatulas have no recipe, mismatched inspection `item` versus receipt `item_id` keys, an incorrect mask field and the wrong budget owner. Native comparison fixtures initially advanced the real episode RNG before the adapter call; they now restore EnvRNG before repeating the action. Initial special-consumable encoder construction failed because native ObservationToken asserts inventory IDs below 58. The fixture builds the encoder before introducing the consumed special item, preserving the native update path. A retained regression proves that an ordinary action with another special consumable still present returns atomic internal_error with unchanged graph and logs.

The native duplicator incremental observation/mask update can differ from a freshly constructed encoder after adding another bench unit. The first fresh-encoder comparison failed and the check was corrected to compare the actual native incremental update path. The guide records both native encoder limitations. No encoder repair or unrelated consumable removal was added. Receipt/copy/catalog/identity checks establish actual equipment success independently of stale masks. Literal Kayn spelling/tracking limits remain documented and unsafe cases reject.

The resumed baseline at `8a47baf6e4c3113824e3e3aae40761f885d80bbd` passed 194 checks with four unchanged whole-lobby repetitions deselected in 95.23 seconds. Its earlier command used incorrect deselection names, selected full-lobby repetitions and exited 143 without a pytest summary; that interrupted attempt is not a pass. The parent's external record `/tmp/tft-mcp-context/resume-11-checks.json` records that correction, 27 native checks, 53 matching installed files and fixed-base scope. A pre-freeze equipment work-in-progress quick run passed 286 checks with the same four deselections in 114.13 seconds. Initial functional head `a76d87fcaffe31d0377e6ee801f16152885c8913` passed 301 checks with the same four deselections in 114.61 seconds. Independent review and the parent found missing invariants despite those passing checks. The resulting corrections required the final frozen-head refresh below.

At exact final functional head `b0cdc8fc0aeb6c0fa6f7efe8d1c1e37f5ca0ee0b`, run from the writer worktree:

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests -k 'not seed_zero_full_lobby and not real_seed_zero and not terminal_inspection and not terminal_public_list' -q
```

Passed 314 checks with four unchanged whole-lobby repetitions deselected in 115.79 seconds. This includes all 120 focused equipment checks. The working diff contained only this evidence record; source, tests, configuration and the accepted Spec were exactly the frozen functional head.

Affected unchanged native checks ran from `/tmp/tft-mcp-context/issue-11-simulator-runtime` with the extension source selected explicitly:

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=/tmp/tft-mcp-worktrees/issue-11/mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest /tmp/tft-mcp-worktrees/issue-11/UnitTests/rng_test.py /tmp/tft-mcp-worktrees/issue-11/UnitTests/player_test.py /tmp/tft-mcp-worktrees/issue-11/UnitTests/action_space_test.py /tmp/tft-mcp-worktrees/issue-11/UnitTests/step_function_test.py -q
```

All 27 passed at the initial functional head in 0.72 seconds, after the first corrective head in 0.75 seconds, and at the final functional head in 0.68 seconds. Native source remains unchanged. An external-cwd import check resolved Simulator under `/tmp/tft-mcp-env/lib64/python3.14/site-packages/Simulator` and confirmed all 53 installed Python files byte-match the unchanged checkout. A first identity diagnostic executed from the writer root resolved the checkout; only the corrected external-cwd diagnostic establishes installed-native identity. The CPU environment remains Python 3.14.7 with MCP 1.30.0 and the versions recorded in CONTINUATION.md. No shared package was reinstalled.

Both scope commands and whitespace checks passed at the functional head:

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python mcp/server/scripts/check_scope.py 8a47baf6e4c3113824e3e3aae40761f885d80bbd
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python mcp/server/scripts/check_scope.py da1a926784ff40d24bd374cd40ae7fb536982eeb
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python mcp/server/scripts/check_scope.py 54cbb8bb9e9933a80ea04bf99989bf001fe4774e
git diff --check 8a47baf6e4c3113824e3e3aae40761f885d80bbd
git diff --stat 33c2c6e -- Simulator UnitTests pyproject.toml setup.py setup.cfg requirements.txt .github
```

Final scope is nine changed paths against original review base 8a47baf, eight against the final reviewed contract checkpoint da1a926, and 32 against the fixed implementation base. Original-base scope includes the merged parent Spec and supplemental evidence. The writer source/guide changes are seven paths against the final checkpoint; this evidence record is the eighth. Simulator, UnitTests, root packaging/dependencies and CI comparison is empty. The local-only contract has no GitHub CI checks configured. Independent review and actual integration remain parent-owned and are not claimed by this writer.

Four unchanged completed-lobby repetitions were deliberately not rerun. Whole-milestone full-game/all-family replay, final noneditable extension installation and actual Codex/Claude host gameplay remain #12/#13 acceptance. Existing broader Gymnasium baseline failures remain unchanged. Only #11 was implemented; no dependent slice, push, issue/PR mutation, merge into integration, upstream change or worktree cleanup occurred. The parent contract integration was merged into this slice only.

### Independent review and verified integration

On 2026-10-10 parallel independent Standards and Spec reviewers checked final clean PR head `39fe0674db594ab492a83b44e7fe8c9eb5fa51f8` against `da1a926784ff40d24bd374cd40ae7fb536982eeb`. Both axes reported zero remaining findings. The Spec reviewer independently passed all 120 equipment/SDK checks in 22.77 seconds and verified both original P2 corrections, native trait timing and the early-board-cascade guard. The last source-to-PR delta contains verification prose only.

The separate merger independently passed 314 checks with the four documented deselections in 112.50 seconds, 27 affected native checks in 0.86 seconds and the external-cwd installed-source comparison of 53 identical Python files. It merged [PR #23](https://github.com/KyleDerZweite/TFTMuZeroAgent/pull/23) only into the fork integration branch at `929658a608b9636ddf5271e46f731aa406ffe681`, with exact parents `da1a926` and `39fe0674` and full tree `a1db758e56e769d84b7e6fa959830c971228582e` identical to the tested artifact. At the actual merge, all 120 equipment checks passed again in 22.94 seconds. Complete scope checks passed for eight slice paths and 32 fixed-base paths, with whitespace and native/root/dependencies/CI comparisons clean before and after merge. The remote integration tip and the clean root fast-forward were verified. No configured CI was claimed. The tracker acceptance was then updated from this evidence and #11 closed. The next dependent slice #12 entered contract review only after this verified integration.

## Complete-game acceptance slice #12

All 322 extension checks and 29 relevant unchanged native checks passed at frozen functional head `528ff4fee9827f699ac223045483df93f54e2781`. Independent three-process production replay completed with identical action receipts, category checkpoints, ordered baseline progression, configuration and placements.

Implementation and local verification started on 2026-10-10 in `/tmp/tft-mcp-worktrees/issue-12`, branch `feat/mcp-12-acceptance`, from clean reviewed checkpoint `6c8da62f6453d798228d57ad2054f780f84f76f1`. Before editing, the writer verified this workspace, branch, HEAD, empty Git status and fork origin `https://github.com/KyleDerZweite/TFTMuZeroAgent.git`. Native REST blockers #4/#5/#9/#10/#11 were closed and integrated, including #11 merge `929658a608b9636ddf5271e46f731aa406ffe681`. The accepted source procedure is [the Spec's complete-game contract](../SPEC.md#complete-game-acceptance-contract), reviewed against that integration and published at the checkpoint. The read-only detailed review is `/tmp/tft-mcp-context/design-12-current.md`.

Frozen functional head is `528ff4fee9827f699ac223045483df93f54e2781`. It adds only `tests/test_acceptance.py`. Production transport, session, transaction, scheduler, budget, RNG and catalog owners remain unchanged. The final evidence commit changes this document only. Tests and their recorded results belong to the functional head, reviewed base and empty functional working diff. The source was frozen before the final suite partitions and native checks.

The new official SDK production-stdio acceptance records a naturally legal seed-zero full game, then consumes its exact successful action tape in two fresh server processes. The third adds repeated/reordered category and rule reads plus an occupied same-location movement rejection. Every run successfully exercises all 25 discovered tools. All singular action receipts, checkpoints after each action, complete recorded configuration/runtime/source identity, ordered native progression, terminal outcome and all eight placements are compared. Only copied generated game IDs and separately validated transport IDs/native paths are excluded from equality. Gameplay values, nested equipment/Chosen/form/linkage fields, action repetitions and list order remain exact.

The natural prefix ends planning, buys actual shop slots 0 and 1, ends planning, moves an owned board unit, equips its naturally received sparring_gloves, sells the other benched unit, and uses earned gold for refresh and XP. Subsequent end_turn calls reach lobby completion with the unchanged seven Default_Agent(False) opponents. Target selection occurs only in the original run; replay processes never select replacement offers or locations. The 40-decision ceiling retains partial tape, receipts, checkpoints and SDK transcript if completion fails. Terminal checks require frozen own categories, eliminated-player versus final-lobby rounds, all completed public players, rejected actions/start, and outcome-preserving close.

Separate bounded production scenarios prove fourteen legal board moves exhaust the public budget without combat, the fifteenth rejects unchanged, and explicit end_turn remains possible. Another checks incomplete explicit close, idempotent idle close, matching seeded restart, implicit shutdown's game_closed/closed_incomplete record and a fresh idle process. The audit-unavailable scenario captures the external SDK log_unavailable result, preserves accepted audit/native bytes and all public categories, then restores the destination and successfully retries. New failed-candidate native directories remain diagnostic artifacts. They are not accepted progression or proof of persistent errors in an unavailable audit destination.

Each preserved raw JSONL audit is checked against the ordered SDK transcript, including exact request arguments, structured results, error flags and contiguous server sequences. Successful calls have one request/result pair with permitted lifecycle/progression records between them. Rejections have the accepted combined tool_error representation and no committed candidate progression. Every referenced accepted native directory is below that run's explicit native root and contains log.txt. Completed-lobby and close outcomes correspond to the terminal SDK result. Raw native files remain available; no broad timestamp or numeric log rewriting is used.

The shared CPU environment remains `/tmp/tft-mcp-env`, Python 3.14.7, MCP 1.30.0, NumPy 2.5.3, PettingZoo 1.27.0 and Gymnasium 1.4.0. No shared package was reinstalled. The unchanged simulator was installed noneditably from archived revision `f4b194c8fe939bff46d2d57b75ce610fbda93ddf`, source archive `/tmp/tft-mcp-context/simulator-source-10`. Final commands pass this operator revision through TFT_MCP_SIMULATOR_REVISION; the acceptance test also validates the supported sha256 identity when no override is supplied. Recorded installed-source digest is `43db488008ac7503201978bef81230635d65a7d15c37b82b6cd368d2fa669dcb`. The independent external-cwd import resolves `/tmp/tft-mcp-env/lib/python3.14/site-packages/Simulator`, and all 53 installed Python files byte-match the writer checkout. Exact import/digest/head/base evidence is `/tmp/tft-mcp-context/issue12-installed-identity.json`.

Final extension commands run from external writable cwd `/tmp/tft-mcp-context/issue12-frozen-runtime`, using absolute extension source, test and configuration paths. Production subprocesses also launch from their own external temporary directories through the unchanged `test_protocol.client`. This establishes actual installed-native integration; it is not #13's fresh extension installation or actual Codex/Claude connection evidence.

```sh
env -u APPIMAGE PYTHONHASHSEED=0 TFT_MCP_SIMULATOR_REVISION=f4b194c8fe939bff46d2d57b75ce610fbda93ddf PYTHONPATH=/tmp/tft-mcp-worktrees/issue-12/mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c /tmp/tft-mcp-worktrees/issue-12/mcp/server/pyproject.toml /tmp/tft-mcp-worktrees/issue-12/mcp/server/tests -k 'not seed_zero_full_lobby and not real_seed_zero and not terminal_inspection and not terminal_public_list and not three_process_replay' -q --basetemp=/tmp/tft-mcp-context/issue12-frozen-A

env -u APPIMAGE PYTHONHASHSEED=0 TFT_MCP_SIMULATOR_REVISION=f4b194c8fe939bff46d2d57b75ce610fbda93ddf PYTHONPATH=/tmp/tft-mcp-worktrees/issue-12/mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c /tmp/tft-mcp-worktrees/issue-12/mcp/server/pyproject.toml /tmp/tft-mcp-worktrees/issue-12/mcp/server/tests/test_inspection_protocol.py::test_terminal_inspection_retains_elimination_round_until_close /tmp/tft-mcp-worktrees/issue-12/mcp/server/tests/test_progression.py::test_real_seed_zero_elimination_finishes_lobby_and_freezes_state /tmp/tft-mcp-worktrees/issue-12/mcp/server/tests/test_protocol.py::test_seed_zero_full_lobby_read_independent_replay /tmp/tft-mcp-worktrees/issue-12/mcp/server/tests/test_public_inspection_protocol.py::test_terminal_public_list_and_removed_winner_preserve_own_categories -q --basetemp=/tmp/tft-mcp-context/issue12-frozen-B
```

Partition A passed 317 checks with exactly five deselections in 121.72 seconds. Its log is `/tmp/tft-mcp-context/issue12-frozen-A.log`; new bounded SDK transcripts/audits/native files are under `/tmp/tft-mcp-context/issue12-frozen-A/`. Partition B runs all four previously deferred complete-lobby nodes. Partition C is the new exact-tape three-process test and is independently executed at the frozen head by the Spec reviewer. Exact collection manifests `/tmp/tft-mcp-context/issue12-collected-{all,A,B,C}.txt` and `/tmp/tft-mcp-context/issue12-collection-partition.json` prove a disjoint union of 317 + 4 + 1 = 322 collected tests, with no omitted node. The passed B and C results are recorded below.

Relevant unchanged native checks ran from `/tmp/tft-mcp-context/issue12-native-runtime`:

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=/tmp/tft-mcp-worktrees/issue-12/mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest /tmp/tft-mcp-worktrees/issue-12/UnitTests/rng_test.py /tmp/tft-mcp-worktrees/issue-12/UnitTests/player_test.py /tmp/tft-mcp-worktrees/issue-12/UnitTests/action_space_test.py /tmp/tft-mcp-worktrees/issue-12/UnitTests/step_function_test.py /tmp/tft-mcp-worktrees/issue-12/UnitTests/bench_full_repro_test.py -q
```

All 29 passed in 1.27 seconds, comprising the established 27 RNG/Player/action/step checks and both bench checks. Raw summary is `/tmp/tft-mcp-context/issue12-frozen-native.log`. An initial development command named nonexistent bench_overflow_test.py and collected no tests, exit 4; the corrected final command above is the passed native evidence.

Before evidence prose, both scope commands passed, covering one changed slice path and 33 fixed-base paths. Whitespace and unchanged Simulator/UnitTests/root packaging/dependency/CI comparisons passed, and the functional working tree was clean:

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python mcp/server/scripts/check_scope.py 6c8da62f6453d798228d57ad2054f780f84f76f1
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python mcp/server/scripts/check_scope.py 54cbb8bb9e9933a80ea04bf99989bf001fe4774e
git diff --check 6c8da62f6453d798228d57ad2054f780f84f76f1
git diff --exit-code 33c2c6e -- Simulator UnitTests pyproject.toml setup.py setup.cfg requirements.txt .github
git status --short
```

Development used TDD at the accepted SDK/audit seam. The ordered audit validator first raised NotImplementedError against a real start/close transcript, then passed. The short natural prefix reached all action families; its first assertion incorrectly used xp instead of public economy exp and was corrected. The audit failure check initially compared newly retained diagnostic candidate files as though they were accepted files; it now verifies every original accepted file remains identical. Those were test-assumption failures, not production defects. The separate prefix test and its development-only branch were removed before freeze because final full acceptance includes their assertions.

The first complete original game finished, but the later parent-process source check incorrectly required a root-origin pytest import to resolve outside the checkout. That attempt failed after 67.16 seconds and did not establish replay. Independent installed-file identity now reads distribution.locate_file, and final commands use external cwd. The subsequent development three-process run passed in 205.52 seconds; its raw artifacts are `/tmp/tft-mcp-context/issue12-full-dev2/test_production_full_game_exac0/{0,1,2}` and its summary is `/tmp/tft-mcp-context/issue12-full-dev2.log`. It used a previously loaded work-in-progress helper, so it is development evidence, not the final frozen-head C result. A duplicate writer C launch was stopped with exit 143 after the parent assigned C to the independent reviewer. Its partial audit remains under `/tmp/tft-mcp-context/issue12-frozen-C/` and is not claimed as passed.

Existing unsafe native glove/trait-origin/Kayn/early-board-duplicate cases remain explicitly unsupported, as documented in the Equipment contract and guide. Inherited focused real-session and official SDK memory-stream scenarios establish supported rare mechanisms without claiming natural production acquisition. The unchanged broader Gymnasium tests test_gymnasium_item_env and test_gymnasium_single_player_env retain their previously recorded failures under Gymnasium 1.4.0; they were not rerun or repaired here. Only the listed 29 unchanged native checks were executed. Fresh noneditable extension installation, active Codex/Claude host connections and the required real Codex LLM full game remain #13 obligations. No target release, push, PR/issue mutation, merge, upstream change or worktree cleanup occurred in this writer.

The requirements matrix below ties inherited scenarios to their final-head suite partitions. The read-only audit `/tmp/tft-mcp-context/acceptance-matrix-12.md` contains exact collected parameter variants and limits. Its original 318-node collection is historical; the final 322-node manifest and executed A/B/C partition results establish current execution.

| Accepted requirement | Current test nodes, relative to mcp/server | Evidence |
| --- | --- | --- |
| Complete legal game, all 25 tools, exact three-process replay, read/rejection independence, ordered audit/configuration/progression/outcome | `tests/test_acceptance.py::test_production_full_game_exact_tape_three_process_replay` | C, official SDK production stdio and unchanged installed simulator. |
| Public budget capacity, unchanged round, rejected fifteenth action and reserved end_turn | `tests/test_acceptance.py::test_production_fourteen_legal_moves_reserve_explicit_end_turn`; `tests/test_progression.py::test_reserved_budget_never_starts_combat` | A, legal SDK moves plus concrete reserve-owner checks. |
| Incomplete close, implicit shutdown, idempotence and seeded fresh restart | `tests/test_acceptance.py::test_production_incomplete_close_shutdown_and_restart`; `tests/test_protocol.py::test_strict_inputs_lifecycle_and_restart` | A, actual process exit and raw game_closed reasons. |
| Audit-unavailable external errors, preserved accepted bytes and recovery | `tests/test_acceptance.py::test_production_unavailable_audit_preserves_external_error_and_recovers`; `tests/test_protocol.py::test_audit_and_native_failure_leave_idle` | A, actual unavailable filesystem destination. External errors are retained separately from accepted JSONL. |
| Startup/native/audit failures and diagnostic retention | `tests/test_session.py::test_failed_initialization_restores_rng_and_retains_diagnostic_evidence`; `::test_final_audit_failure_rolls_back_started_game`; `::test_startup_audit_durability_failure_identifies_unpublished_diagnostics`; `tests/test_progression.py::test_unavailable_audit_is_rejected_before_native_construction` | A, real adapter with filesystem/internal boundary failures. |
| Full graph identities/aliases, episode/process RNG, native/audit bytes, fresh-reference retry across native/observation/mask/baseline/receipt/postcondition/log failures | `tests/test_movement.py::test_movement_failure_rolls_back_full_graph_rng_logs_and_fresh_retry[native_action]` through `[audit]`; `tests/test_equipment.py::test_equipment_failure_rolls_back_full_graph_rng_logs_and_fresh_retry[native_action-thieves_gloves]` through `[audit-reforger]` | A, all ten movement and twenty randomized equipment variants. Entire pickled graph equality, owner identities, process bindings and retry are asserted. |
| Buy/sell, refresh/XP and end_turn failure atomicity, late result/close commit failure | `tests/test_buy_sell.py::test_failed_action_discards_aggregate_rng_logs_and_retries[buy-native_action]` through `[sell-audit]`; `tests/test_shop_xp.py::test_failed_action_discards_aggregate_rng_logs_and_retries[refresh_shop-native_action]` through `[buy_xp-audit]`; `tests/test_progression.py::test_failed_candidate_restores_graph_logs_rng_and_retries`; `::test_late_transport_result_failure_and_failed_close_keep_committed_game`; `tests/test_protocol.py::test_failed_close_keeps_game_active` | A, fourteen buy/sell and sixteen refresh/XP variants plus progression/close faults. These families prove their actual gameplay/RNG/owner/alias/log assertions; stronger entire-graph equality belongs to the preceding row. |
| Invalid schema/gameplay requests and actionable errors | `tests/test_protocol.py::test_strict_inputs_lifecycle_and_restart`; `tests/test_movement.py::test_production_strict_movement_inputs_preserve_state[asyncio]`; `tests/test_buy_sell_protocol.py::test_stdio_strict_requests_rejections_and_recovery`; `tests/test_shop_xp.py::test_rejections_preserve_gameplay_and_follow_validation_order[refresh_shop]` and `[buy_xp]` | A, native SDK errors and concrete legality order. C adds occupied same-location movement rejection to full replay. |
| Terminal snapshots, controlled elimination versus final round, completed placements/winner and close | `tests/test_inspection_protocol.py::test_terminal_inspection_retains_elimination_round_until_close`; `tests/test_progression.py::test_real_seed_zero_elimination_finishes_lobby_and_freezes_state`; `tests/test_protocol.py::test_seed_zero_full_lobby_read_independent_replay`; `tests/test_public_inspection_protocol.py::test_terminal_public_list_and_removed_winner_preserve_own_categories` | B, all four inherited complete-lobby scenarios. C additionally checks the all-family tape's terminal and close receipts. |
| Category privacy/read purity and installed catalog consistency/RNG preservation | `tests/test_inspection.py::test_read_purity_preserves_entire_game_rng_caches_aliases_and_later_progression`; `tests/test_public_inspection.py::test_public_reads_preserve_complete_graph_rng_caches_and_bindings`; `tests/test_item_catalog.py::test_catalog_queries_preserve_definitions_and_rng_for_every_item`; `tests/test_champion_catalog.py::test_all_catalog_definitions_are_source_consistent_and_reads_are_pure` | A, whole native graphs and every installed catalog definition. C adds meaningful own/living-opponent category and catalog reads. |
| Supported rare native mechanics and explicit unsafe cases | `tests/test_equipment_protocol.py::test_sdk_memory_rare_equipment_native_fixture` in its six collected variants; `tests/test_movement.py::test_sdk_memory_rare_real_azir_fixture_and_atomic_guard_rejection[asyncio]`; `tests/test_shop_xp.py::test_sdk_memory_real_session_cap_success_and_atomic_rejection[asyncio]`; inherited concrete equipment/movement/buy-sell corruption and safety scenarios | A, official SDK memory streams with real native fixture sessions. Combination/remover/reforger/duplicator/gloves/literal board Kayn/Azir/cap evidence remains separate from natural production acquisition. |

The independent Spec reviewer ran partition C in clean detached `/tmp/tft-mcp-worktrees/review-12-spec` at exact functional head `528ff4fee9827f699ac223045483df93f54e2781`, using external cwd `/tmp/tft-mcp-context/review-12-spec-runtime`:

```sh
env -u APPIMAGE PYTHONHASHSEED=0 TFT_MCP_SIMULATOR_REVISION=f4b194c8fe939bff46d2d57b75ce610fbda93ddf PYTHONPATH=/tmp/tft-mcp-worktrees/review-12-spec/mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c /tmp/tft-mcp-worktrees/review-12-spec/mcp/server/pyproject.toml /tmp/tft-mcp-worktrees/review-12-spec/mcp/server/tests/test_acceptance.py::test_production_full_game_exact_tape_three_process_replay -q --basetemp=/tmp/tft-mcp-context/review-12-spec-tests --junitxml=/tmp/tft-mcp-context/review-12-spec-junit.xml
```

Partition C passed one test in 191.37 seconds, exit 0. Every process completed the same 20-action tape, including 13 end_turn calls and all seven singular gameplay families, with exactly 2,511 ordered native progression records. Original/replay/extra-read processes made 307/307/494 tool calls and produced 3,108/3,108/3,481 audit events. Each succeeded with all 25 tools and reached player_0 placement 8, lobby_complete true and reason lobby_complete. All placements matched exactly: player_0=8, player_1=4, player_2=5, player_3=1, player_4=7, player_5=2, player_6=3, player_7=6. Own terminal categories retain round 14 while final lobby status/get_round is 25. Final audit, category, receipt, source/configuration and ordered progression comparisons passed.

Independent evidence is `/tmp/tft-mcp-context/review-12-spec-junit.xml`, `/tmp/tft-mcp-context/review-12-spec-runtime-summary.json` and `/tmp/tft-mcp-context/review-12-spec-tests/test_production_full_game_exac0/{0,1,2}`. The reviewer remained clean at the frozen head. Independent functional Standards and Spec reviews both reported zero remaining findings, in `/tmp/tft-mcp-context/review-12-standards.md` and `/tmp/tft-mcp-context/review-12-spec.md`. Final evidence-prose review and integration remain parent-owned and are not claimed by this writer.

Partition B passed all four inherited completed-lobby checks in 261.57 seconds, exit 0. Raw summary is `/tmp/tft-mcp-context/issue12-frozen-B.log`, with real-lobby artifacts under `/tmp/tft-mcp-context/issue12-frozen-B/`. Together A317, B4 and independently reviewed C1 passed every node of the exact 322-test collection. No extension check is failed, omitted or unexecuted at this functional head. Historical development failures and the separately unexecuted broader native/host checks remain explicitly distinguished above.

After adding only this evidence prose, final slice scope passed for two paths and fixed-base scope for 33 paths. Staged and unstaged whitespace checks, untracked-path inclusion and unchanged native/root/dependency/CI comparisons passed. The evidence commit contains only verification.md; production/test source is byte-identical to the frozen functional head.

### Independent final review and verified integration

Both independent reviewers safely refreshed to final clean PR head `b23a8d38a35e795478dd723ff8f5675e779622d0` and reviewed its complete two-file diff against `6c8da62f6453d798228d57ad2054f780f84f76f1`. Standards and Spec each reported zero remaining findings. The final evidence-only delta matched the raw A/B/native summaries, exact collected-node manifest and independent C artifacts. The disjoint partition union was independently verified against all 322 collected nodes.

The separate merger independently inspected those manifests/results, installed source identity and raw three-process tape/progression/outcome evidence, then passed the three new bounded SDK acceptance tests in 11.29 seconds. It guarded the merge of [PR #24](https://github.com/KyleDerZweite/TFTMuZeroAgent/pull/24) into the fork integration branch at `e87ec87ee27df88cea2ca42bdb597d08e8437366`, verified exact parents `6c8da62` and `b23a8d3`, full tested tree identity and the fetched fork remote tip. On the actual merge, all three bounded acceptance tests passed again in 11.26 seconds. Slice/fixed scope, whitespace and unchanged Simulator/UnitTests/root packaging/dependencies/CI checks passed. No configured CI was claimed. The clean root was fast-forwarded, and #12 acceptance updated and closed from the integrated evidence. Dependent #13 entered its installation/client contract review only afterward.

## Issue #13 archived installation and native host evidence

Verified on 2026-10-10 in the isolated writer `/tmp/tft-mcp-worktrees/issue-13`, branch `feat/mcp-13-client-setup`, fork origin `KyleDerZweite/TFTMuZeroAgent`. The reviewed base is `51fa6c87220854da0581321909dbf9f153dc6c3a`, following verified #12 merge `e87ec87ee27df88cea2ca42bdb597d08e8437366`. The clean functional head is `ebc70e105c42dfd47e2b13c63d00193d72399a94`. It changes operational documentation and the existing discovery assertion only; production code is unchanged. This final documentation commit records later evidence and corrects the external Claude project-root instruction. Its head is the containing commit, available through `git rev-parse HEAD`; the installed source remains the exact functional head above.

The persistent operator-owned resource root is `/home/kyle/.local/share/tft-mcp/issue-13-20261010`. Its fresh venv, source archive, external host cwd and separate SDK/Codex/Claude logs remain available. Directory permissions are owner-only at the task root; acceptance files were created with umask 077. No credentials or raw operator configuration were copied into the repository or shared temporary files.

The immutable `source.tar` was produced with `git archive` from the functional head. Its SHA-256 is `8af572199aa7010bc066116f1c16f467a7740c864bfb91256306f6afa5d5a3ce`. Both unchanged simulator and separate extension were installed noneditably from the extracted archive, in that order. Both source revisions are `ebc70e105c42dfd47e2b13c63d00193d72399a94`; both distribution versions are `0.1.0`. The installed simulator digest is `43db488008ac7503201978bef81230635d65a7d15c37b82b6cd368d2fa669dcb`, matching the verified unchanged core. NumPy 2.5.3, PettingZoo 1.27.0, Gymnasium 1.4.0 and MCP 1.30.0 match the prior verified runtime. Python is CPython 3.14.7. `pip check` passed with no broken requirements. No GPU, training or model-hosting dependencies were installed.

External-cwd imports with APPIMAGE and PYTHONPATH removed resolve `Simulator` and `tft_mcp` inside `venv/lib64/python3.14/site-packages`. The absolute production launcher is `/home/kyle/.local/share/tft-mcp/issue-13-20261010/venv/bin/tft-mcp`. Installed bootstrap audit records the archive revision, actual digest, dependency versions, unchanged configuration, eight players, baseline identity, fixed hash seed 0 and hash probe `3581204761862787471`. Identity evidence is in `logs/sdk/identity.json` and `logs/sdk/start-identity.json`; build logs are `logs/sdk/install-simulator.log` and `logs/sdk/install-extension.log`.

The focused changed discovery assertion passed with 1 test and 6 deselected before freezing. The installed official SDK protocol command below passed 6 tests and deselected the existing full-lobby replay. It covers exact discovery of all 25 unique tools, idle status, strict input errors, lifecycle and clean restart, deterministic bootstrap, audit/native startup failures, failed-close recovery and JSON-only protocol stdout with native malformed-envelope errors. Results are `logs/sdk/protocol.log`; actual audit/native subprocess artifacts are under `logs/sdk/protocol/`.

```sh
cd /home/kyle/.local/share/tft-mcp/issue-13-20261010/host
env -u APPIMAGE -u PYTHONPATH TFT_MCP_SIMULATOR_REVISION=ebc70e105c42dfd47e2b13c63d00193d72399a94 TFT_MCP_TEST_COMMAND=/home/kyle/.local/share/tft-mcp/issue-13-20261010/venv/bin/tft-mcp /home/kyle/.local/share/tft-mcp/issue-13-20261010/venv/bin/python -m pytest -c /home/kyle/.local/share/tft-mcp/issue-13-20261010/source/mcp/server/pyproject.toml /home/kyle/.local/share/tft-mcp/issue-13-20261010/source/mcp/server/tests/test_protocol.py -k 'not full_lobby' -q --basetemp=/home/kyle/.local/share/tft-mcp/issue-13-20261010/logs/sdk/protocol
```

No production change or runtime dependency change justifies repeating #12's 322 extension tests, 29 native checks or three-process complete-game replay. Their integrated evidence remains above. Historical Gymnasium failures remain unmodified and were not rerun. Local scope checks against the slice base and fixed implementation base include committed, staged, unstaged and untracked paths. No Simulator, UnitTests, root package/dependency or CI changes are present. Final scope checks passed for 7 changed paths versus the slice base and 34 versus fixed base `54cbb8bb9e9933a80ea04bf99989bf001fe4774e`. `git diff --check`, local guide links and prose dash-punctuation checks passed. No applicable GitHub CI was run or claimed.

### Native registration and preservation

Actual installed hosts are Codex CLI 0.162.1 and Claude Code 2.1.294. Native authentication checks observed ChatGPT login for Codex and firstParty api_key_helper login for Claude. The protected stores remain operator-owned references; credential values were never printed. Both `tft` entries use `/usr/bin/env` with `-u APPIMAGE` and the absolute installed launcher, plus explicit separate audit/native paths and `TFT_MCP_SIMULATOR_REVISION` equal to the archive revision.

The first registration process held original parsed configuration in memory, but stopped after a native Claude add returned nonzero after writing. Its initial preservation assertion was unexecuted and cannot be claimed. The native Claude local scope initially resolved to `/home/kyle` because of that ancestor's existing empty `.git` marker. Only this task's known-new entry was removed through native local scope. An empty external Git project now establishes `/home/kyle/.local/share/tft-mcp/issue-13-20261010/host` as Claude's exact private local project root. No repository client configuration or host permission setting was created.

The authorized successful rerun removed only the known-new task entries, retained the full fresh semantic baseline in memory, inspected named absence, and added each native entry again. Parsed Codex equality excluded only `mcp_servers.tft`; Claude equality excluded only `projects[exact_external_cwd].mcpServers.tft`. All unrelated/preexisting configuration compared equal. Existing Claude settings and Codex auth files compared byte-identical before/after. Codex retained approval_policy `never` and sandbox_mode `workspace-write`; Claude retained permission defaultMode `auto` with no added allow, deny or ask entries. No approval, allowlist or sandbox override was supplied.

The prior external-project creation added only native default metadata keys `allowedTools`, `disabledMcpjsonServers`, `enabledMcpjsonServers`, `hasClaudeMdExternalIncludesApproved`, `hasClaudeMdExternalIncludesWarningShown`, `hasTrustDialogAccepted` and `mcpContextUris`. The correction comparison retained every existing home-project field and reported those additions separately, rather than excluding the whole project. No authentication or permission configuration was added. After the successful rerun, repeated native get inspections retained already-correct entries without any semantic rewrite. Sanitized proof is `logs/registration-preservation.json`; named-entry readbacks are `logs/codex/registration.txt` and `logs/claude/registration.txt`.

### Native Codex acceptance remains open

A fresh real native CLI session used exactly `codex exec --model gpt-6.1-sol -c 'model_reasoning_effort="low"' --json --skip-git-repo-check -`, with the separate documented game prompt on stdin and the external host cwd. No permission flags or custom runner were used. Native thread ID is `01a122d1-21a3-7ae0-878a-0273473cf716`. Protected native rollout metadata confirms model `gpt-6.1-sol`, effort `low`, approval `never`, workspace-write sandbox and the external cwd; its sanitized reference is `logs/codex/native-context.json`. The native JSONL transcript records an actual individual `tft/get_game_status {}` call, followed by the exact host error `MCP tool call requires approval, but approval policy is never`.

The host rejected the call before the server executed it. No game was started, and no game/server correlation ID, eight placements, terminal inspection or close receipt exists for this native run. The transcript is `logs/codex/session.jsonl`, diagnostics are `logs/codex/session.stderr`, and the supplied prompt is `logs/codex/game-prompt.txt`. The model's own result file is `host/tft-acceptance-result.json`; it is supplementary and does not replace the actual failed tool result. Required real Codex full-game acceptance remains open. Installed SDK success cannot satisfy it.

Read-only investigation checked the current [official configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference) and [native upstream MCP approval source](https://github.com/openai/codex/blob/main/codex-rs/core/src/mcp_tool_call.rs). The source contains the exact never-policy rejection. The documented target-only `mcp_servers.tft.default_tools_approval_mode = "approve"` is a concrete potential remedy while retaining global never/workspace-write, but it changes tool approval policy and conflicts with the current preservation contract. It was not applied. Correct read-only annotations could describe queries; they cannot honestly authorize mutating game actions or establish the required full game. No server annotations or production behavior were changed to bypass the host decision. A policy decision outside this implementation remains necessary before that proposed remedy can be executed.

### Native Claude gameplay remains unavailable

Fresh normal native `claude --print --output-format stream-json --verbose` sessions used the external cwd and meaningful authorized short gameplay prompt. Their native init records list all 25 `mcp__tft__` tools and `tft` connected with local source, under permissionMode `auto`. This proves active startup/discovery, but not execution of an idle or gameplay call.

The configured default `claude-opus-4-6` returned HTTP 404 `model_not_found` before tools could execute. A normal per-session `--model sonnet` resolved `claude-sonnet-5-5` with the same failure; exact `claude-sonnet-4-6` also failed. Sanitized settings inspection found the configured small-model identifier `claude-haiku-4-5-20251001`, so one final bounded per-session attempt used it and also returned HTTP 404 `model_not_found`. Authentication, provider endpoint and global model settings were unchanged. Further identical retries are not useful evidence.

The final exact-project default session is `fa433c6c-a619-4cb2-bf10-b284b6ef5a35` in `logs/claude/session-exact-cwd.jsonl`. The configured-Haiku session is `2ebc6906-988a-4358-8042-d891790f5a94` in `logs/claude/session-configured-haiku.jsonl`. Earlier attempts and their stderr files remain in the same protected host log directory. No Claude game, actual idle result or cleanup receipt was produced. Actual Claude tool execution/gameplay is unavailable until its configured authenticated provider offers an accessible model. This separate limitation does not replace or close required Codex acceptance.

Installation and operational documentation are implemented and locally verified. Registration preservation passed on the successful rerun. Native gameplay acceptance is incomplete for the exact observed reasons above. No release, PR/issue mutation, push, merge, upstream change or resource cleanup was performed by this writer.
