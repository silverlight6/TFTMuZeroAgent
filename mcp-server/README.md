# TFT simulator MCP server

This directory owns the local Python MCP extension for the existing TFT Set 4 simulator. The simulator remains a separately installed, unchanged dependency. An external MCP client owns model inference.

Implementation has not started. The production launcher, package configuration, installation commands, and runnable checks will be introduced by [issue #2](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/2). There is no server to run yet.

Read [SPEC.md](SPEC.md) for the accepted behavior, module responsibilities, lifecycle interface, scope constraints, and test strategy. [Milestone #1](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/1) is its published tracker mirror. GitHub owns ticket status, blocking relationships, and execution order. The root [glossary](../GLOSSARY.md) owns domain terminology.

For a slice, read its ticket and relevant Spec contract, start from the integrated prerequisite revision, implement the smallest complete behavior, and verify through the production MCP interface. Keep code, dependencies, tests, commands, and operational documentation in this directory. Simulator field access belongs in the concrete adapter; the transport contains no game rules.

Use ordinary Python functions and concrete classes with clear names and short control flow. Explain the reason for simulator-specific recovery or scheduling code. Add abstraction only when an existing caller or acceptance scenario needs it.

Local verification includes a real SDK client over stdio, focused adapter failure tests, relevant existing simulator checks, and a scope check against the reviewed base. Installation acceptance also starts the installed server from outside the checkout. GitHub Actions is not part of the current delivery model.

Slice PRs target the integration branch feat/mcp-server-main. origin is the fork remote. main remains the fork main branch; upstream is the original project remote. Completing a slice does not authorize the final merge or upstream submission.
