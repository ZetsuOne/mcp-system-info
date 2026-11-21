"""Тестовый клиент: запускает server.py по stdio и вызывает get_system_info."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

SERVER_PATH = Path(__file__).resolve().parent / "server.py"


def _print_gpus(payload: dict) -> None:
    gpus = payload.get("gpus") or []
    print(f"\nGPU ({len(gpus)}):")
    if not gpus:
        print("  (не обнаружено)")
        return
    for index, gpu in enumerate(gpus):
        print(f"  [{index}] {gpu.get('name')}  source={gpu.get('source')}")
        print(
            "      util={gpu_util_percent}%  "
            "vram={vram_used_gb}/{vram_total_gb} GB  "
            "temp={temperature_c}°C".format(
                gpu_util_percent=gpu.get("gpu_util_percent"),
                vram_used_gb=gpu.get("vram_used_gb"),
                vram_total_gb=gpu.get("vram_total_gb"),
                temperature_c=gpu.get("temperature_c"),
            )
        )


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Доступные инструменты:", [t.name for t in tools.tools])

            result = await session.call_tool("get_system_info", {})
            print("\nРезультат get_system_info:")

            if result.structured_content is not None:
                payload = result.structured_content
                print(json.dumps(payload, indent=2, ensure_ascii=False))
                if isinstance(payload, dict):
                    _print_gpus(payload)
            else:
                for block in result.content:
                    if isinstance(block, TextContent):
                        print(block.text)
                        try:
                            payload = json.loads(block.text)
                        except json.JSONDecodeError:
                            payload = None
                        if isinstance(payload, dict):
                            _print_gpus(payload)


if __name__ == "__main__":
    asyncio.run(main())
