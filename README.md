English | [Русский](README_RU.md)

# mcp-system-info

Local MCP server that exposes live host telemetry — CPU, RAM, disk, and GPU — to Cursor, Claude Desktop, and any MCP-compatible client.

Built on the official Python `mcp` 2.x SDK (`MCPServer`). A single tool, `get_system_info`, returns a structured snapshot suitable for agents that need to reason about machine load, VRAM headroom, or thermal state before launching heavy workloads.

## Features

- **CPU, RAM, disk** via `psutil` — utilization, free/total capacity in GB
- **Multi-vendor GPU cascade** — vendor-native sources first, OS fallback last
- **Mixed-GPU machines** — NVIDIA + Intel + AMD results are merged without duplicate adapters
- **Safe nulls** — missing temperature or clocks return `null` instead of crashing the server
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
| `source` | `nvidia` \| `intel` \| `amd` \| `windows` |

Collection order:

1. **NVIDIA NVML** — `nvidia-ml-py` (`pynvml`): utilization, VRAM, temperature, clocks
2. **Intel Arc** — `xpu-smi` (Intel oneAPI / IGCL) JSON output when present in `PATH`
3. **AMD** — `amd-smi` / `rocm-smi` JSON; on Windows without those tools, name and capacity via WMI (`Win32_VideoController`)
4. **Windows PDH fallback** — only if every vendor step returned nothing: GPU Engine / Adapter Memory counters via `win32pdh` or PowerShell `Get-Counter`. Temperature is typically unavailable through these APIs → `null`

Vendor results are unioned (name-overlap dedupe). The Windows fallback runs only when the cascade is empty.

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

```json
{
  "cpu_percent": 7.7,
  "disk_free_gb": 157.28,
  "disk_total_gb": 930.1,
  "ram_available_gb": 6.18,
  "ram_total_gb": 15.92,
  "ram_percent": 61.2,
  "gpus": [
    {
      "name": "NVIDIA GeForce RTX 3060",
      "gpu_util_percent": 25.0,
      "vram_used_gb": 1.47,
      "vram_total_gb": 12.0,
      "temperature_c": 51.0,
      "graphics_clock_mhz": 300,
      "memory_clock_mhz": 405,
      "source": "nvidia"
    }
  ]
}
```

## Project layout

| Path | Role |
| --- | --- |
| `server.py` | MCP server (stdio) and `mcp-system-info` entry point |
| `client.py` | Stdio test client that calls `get_system_info` |
| `pyproject.toml` | PEP 621 packaging; console script `mcp-system-info` |
| `requirements.txt` | Same runtime dependencies for classic `pip install -r` |

## License

MIT — see [LICENSE](LICENSE).
