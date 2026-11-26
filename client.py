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


def _print_summary(payload: dict) -> None:
    print(
        f"{payload.get('hostname')}  {payload.get('os')} {payload.get('os_release')}  {payload.get('arch')}"
    )
    print(
        "CPU {cpu_model}  {cpu_percent}%  cores {physical}/{logical}  up {uptime}s".format(
            cpu_model=payload.get("cpu_model"),
            cpu_percent=payload.get("cpu_percent"),
            physical=payload.get("cpu_count_physical"),
            logical=payload.get("cpu_count"),
            uptime=payload.get("uptime_seconds"),
        )
    )
    print(
        "Disk {path}  {free}/{total} GB  {percent}%".format(
            path=payload.get("disk_path"),
            free=payload.get("disk_free_gb"),
            total=payload.get("disk_total_gb"),
            percent=payload.get("disk_percent"),
        )
    )
    warnings = payload.get("warnings") or []
    if warnings:
        print("Предупреждения:")
        for item in warnings:
            print(f"  - {item}")


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
            "temp={temperature_c}°C  "
            "power={power_watts}W  fan={fan_percent}%  limit={power_limit_watts}W".format(
                gpu_util_percent=gpu.get("gpu_util_percent"),
                vram_used_gb=gpu.get("vram_used_gb"),
                vram_total_gb=gpu.get("vram_total_gb"),
                temperature_c=gpu.get("temperature_c"),
                power_watts=gpu.get("power_watts"),
                fan_percent=gpu.get("fan_percent"),
                power_limit_watts=gpu.get("power_limit_watts"),
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

            fast = "--fast" in sys.argv[1:]
            result = await session.call_tool("get_system_info", {"fast": fast})
            print("\nРезультат get_system_info:")

            if result.structured_content is not None:
                payload = result.structured_content
                print(json.dumps(payload, indent=2, ensure_ascii=False))
                if isinstance(payload, dict):
                    _print_summary(payload)
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
                            _print_summary(payload)
                            _print_gpus(payload)


if __name__ == "__main__":
    asyncio.run(main())
