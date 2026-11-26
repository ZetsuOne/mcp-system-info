English | [Русский](README_RU.md)

# mcp-system-info

Local MCP server that exposes live host telemetry — CPU, RAM, disk, and GPU — to Cursor, Claude Desktop, and any MCP-compatible client.

Built on the official Python `mcp` 2.x SDK (`MCPServer`). A single tool, `get_system_info`, returns a structured snapshot suitable for agents that need to reason about machine load, VRAM headroom, or thermal state before launching heavy workloads.

## Features

- **Host, CPU, RAM, disks** — model, core count, uptime, system volume, and every mounted fixed disk
- **Multi-vendor GPU cascade** — vendor tools first, Windows counters fill adapters they missed
- **Mixed-GPU machines** — NVIDIA, Intel, and AMD results are merged. `RTX 3060` and `RTX 3060 Ti` stay separate
- **NVIDIA power telemetry** — watts, fan speed, and power limit when NVML exposes them
- **Safe nulls and warnings** — a missing sensor is `null`. Collection problems are listed in `warnings` instead of crashing the server
- **Short cache** — identical calls within 1.5 seconds reuse the last snapshot
- **Zero-config launch** — run with `uvx` straight from GitHub

## GPU collection cascade

For each detected adapter the server aims to return:

| Field | Meaning |
| --- | --- |
| `name` | Adapter name |
| `gpu_util_percent` | Core utilization (%) |
| `vram_used_gb` / `vram_total_gb` | Used / total VRAM (GB) |
| `temperature_c` | Temperature (°C), or `null` if unavailable |
| `graphics_clock_mhz` / `memory_clock_mhz` | Clocks when the source exposes them |
| `power_watts` / `fan_percent` / `power_limit_watts` | NVIDIA NVML. Other sources leave these `null` |
| `source` | `nvidia` \| `intel` \| `amd` \| `windows` |

Collection order:

1. **NVIDIA NVML** — `nvidia-ml-py` (`pynvml`): utilization, VRAM, temperature, clocks
2. **Intel Arc** — `xpu-smi` (Intel oneAPI / IGCL) JSON output when present in `PATH`
3. **AMD** — `amd-smi` / `rocm-smi` JSON. On Windows without those tools, name and capacity come from the adapter class (`HardwareInformation.qwMemorySize`) and DirectX LUID registry, with WMI only as a last resort
4. **Windows counters** — GPU Engine and Adapter Memory (`Dedicated Usage`, `Dedicated Limit`) via `win32pdh`, or PowerShell `Get-Counter` if PDH is unavailable. Counters are matched to adapters by LUID, not by list position. Temperature is usually unavailable here → `null`

Vendor results are unioned. Windows then adds adapters the vendor steps did not already report, so a laptop iGPU still appears next to a discrete NVIDIA card. `get_system_info` also accepts `fast: true` to skip the blocking CPU sample and the second GPU-counter sample. Utilization may be null in that mode. Memory sizes stay accurate.

## Quick start with `uvx`

Requires [uv](https://docs.astral.sh/uv/). One command, no local clone:

```bash
uvx --from git+https://github.com/ZetsuOne/mcp-system-info mcp-system-info
```

### Cursor / Claude Desktop config

```json
{
  "mcpServers": {
    "system-info": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/ZetsuOne/mcp-system-info", "mcp-system-info"]
    }
  }
}
```

## Classic install (venv)

### Windows (PowerShell)

```powershell
git clone https://github.com/ZetsuOne/mcp-system-info.git
cd mcp-system-info
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation scripts:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Smoke-test:

```powershell
python client.py
```

Run the server alone (stdio; silent until an MCP client connects):

```powershell
python server.py
# or, after `pip install -e .`:
mcp-system-info
```

Local Cursor config (venv path):

```json
{
  "mcpServers": {
    "system-info": {
      "command": "C:\\path\\to\\mcp-system-info\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\mcp-system-info\\server.py"]
    }
  }
}
```

### Linux / macOS (bash)

```bash
git clone https://github.com/ZetsuOne/mcp-system-info.git
cd mcp-system-info
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python client.py
```

Optional vendor tooling for richer GPU fields:

- **NVIDIA** — GPU driver (NVML ships with the driver; `nvidia-ml-py` is already a dependency)
- **Intel Arc** — [Intel XPU System Management Interface](https://www.intel.com/) so `xpu-smi` is on `PATH`
- **AMD** — `amd-smi` (Adrenalin) or `rocm-smi` (ROCm)

Without Intel/AMD utilities the server still works: NVIDIA via NVML; other adapters via the Windows PDH fallback on Windows.

## Example: `get_system_info` response

`disk_*` is the system volume. `disks` lists mounted fixed volumes. Numbers below are one real call on a Windows desktop.

```json
{
  "hostname": "Zetsu",
  "os": "Windows",
  "os_release": "11",
  "arch": "AMD64",
  "cpu_model": "AMD Ryzen 5 5600X 6-Core Processor",
  "cpu_count": 12,
  "cpu_count_physical": 6,
  "cpu_percent": 17.2,
  "uptime_seconds": 30115.6,
  "disk_path": "C:\\",
  "disk_free_gb": 141.62,
  "disk_total_gb": 930.1,
  "disk_percent": 84.8,
  "disks": [
    {"path": "C:\\", "free_gb": 141.62, "total_gb": 930.1, "percent": 84.8, "fstype": "NTFS"},
    {"path": "D:\\", "free_gb": 1.36, "total_gb": 13.4, "percent": 89.9, "fstype": "NTFS"},
    {"path": "F:\\", "free_gb": 582.66, "total_gb": 931.51, "percent": 37.4, "fstype": "NTFS"},
    {"path": "G:\\", "free_gb": 205.01, "total_gb": 931.5, "percent": 78.0, "fstype": "NTFS"}
  ],
  "ram_available_gb": 2.35,
  "ram_total_gb": 15.92,
  "ram_percent": 85.2,
  "gpus": [
    {
      "name": "NVIDIA GeForce RTX 3060",
      "gpu_util_percent": 12.0,
      "vram_used_gb": 2.07,
      "vram_total_gb": 12.0,
      "temperature_c": 55.0,
      "graphics_clock_mhz": 210,
      "memory_clock_mhz": 810,
      "power_watts": 18.3,
      "fan_percent": 0.0,
      "power_limit_watts": 170.0,
      "source": "nvidia"
    }
  ],
  "warnings": []
}
```

## Project layout

| Path | Role |
| --- | --- |
| `server.py` | MCP server (stdio) and `mcp-system-info` entry point |
| `client.py` | Stdio test client. `python client.py --fast` skips blocking samples |
| `pyproject.toml` | PEP 621 packaging and the canonical dependency list |
| `requirements.txt` | Installs this directory, so dependencies come from `pyproject.toml` |
| `tests/` | Parser and GPU-join tests (`pytest`) |

## License

MIT — see [LICENSE](LICENSE).
