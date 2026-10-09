# MCP extension work

Before changing the extension, read [SPEC.md](SPEC.md) and the current slice ticket. Use the root [GLOSSARY.md](../GLOSSARY.md) for domain vocabulary. SPEC.md owns behavior, module responsibilities, style, scope, and verification requirements; update its published milestone mirror when accepted contracts change.

Start from the integrated revision of the ticket's blockers on feat/mcp-server-main. Withdraw readiness from affected tickets when a contract changes, and reconcile dependent work before implementation resumes. A numbered display position is not an additional blocking edge.

Keep new extension code, dependencies, tests, and tooling here. The owner selected local verification only. If simulator evidence shows an unavoidable outside-scope change, record the evidence and minimum proposal before changing the accepted contract. Preserve unrelated work.

Before handing back a slice, verify its actual MCP behavior and failure paths, the affected simulator checks, and the complete changed-path scope against its reviewed base. Record head/base/working diff and distinguish passed, failed, and unexecuted checks. Use the exact local commands documented after they are implemented.

Target slice PRs at feat/mcp-server-main. Do not merge PRs or submit upstream without the owner's authorization. Do not add installation or check commands to documentation as runnable before they exist.
