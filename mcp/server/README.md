# TFT MCP server

This directory will contain the Python MCP server package, its packaging metadata, dependencies, tests, and local developer commands. Read the shared [Spec](../SPEC.md) and [extension entry guide](../README.md) before implementing a slice.

Implementation has not started. [Issue #2](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/2) introduces the production stdio launcher and runnable installation and check commands. The existing simulator is a separately installed, unchanged dependency. Codex and Claude Code are external MCP clients; the server does not host an LLM.
