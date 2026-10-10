# MCP extension work

Before changing the extension, read [SPEC.md](SPEC.md) and the relevant issue or requested change. Use the root [GLOSSARY.md](../GLOSSARY.md) for domain vocabulary. SPEC.md owns behavior, module responsibilities, style, scope and verification requirements. Reconcile affected contracts before implementation.

Keep shared documentation and prompts here. Put server code, packaging, dependencies, tests and tooling in server/. Existing native MCP hosts do not require a custom client package. Simulator source, rules, defaults and root dependencies remain unchanged. If a core change appears unavoidable, record its evidence and minimum proposal before requesting a scope decision. The selected verification model uses local checks only.

Verify actual MCP behavior, failure paths, affected simulator checks and every changed path against the reviewed base. Record head/base/working diff and distinguish passed, failed and unexecuted checks in [server/verification.md](server/verification.md). Developer checks must select the checkout source; installed acceptance must use the production launcher outside the checkout without PYTHONPATH.

Preserve unrelated work and operator configuration. Target the pull request at the branch requested by the owner. Merging, upstream submission and deployment require their own authorization.
