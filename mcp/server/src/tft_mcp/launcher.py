"""Establish interpreter hash configuration before importing server dependencies."""

import os
import sys

HASH_SEED = "0"


def main():
    if os.environ.get("PYTHONHASHSEED") != HASH_SEED:
        environment = dict(os.environ, PYTHONHASHSEED=HASH_SEED)
        os.execve(sys.executable, [sys.executable, "-m", "tft_mcp"], environment)
    import asyncio
    from tft_mcp.transport import serve

    asyncio.run(serve())
