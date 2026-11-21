"""Локальный MCP-сервер: CPU, диск, RAM и GPU (NVIDIA / Intel Arc / AMD / Windows)."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import time
from typing import Any, List, TypedDict
from functools import lru_cache

import psutil
from mcp.server.mcpserver import MCPServer

logger = logging.getLogger(__name__)
mcp = MCPServer("system-info")

_BYTES_PER_GB = 1024**3

class GpuInfo(TypedDict):
    name: str
    gpu_util_percent: float | None
    vram_used_gb: float | None
    vram_total_gb: float | None
    temperature_c: float | None
    graphics_clock_mhz: int | None
    memory_clock_mhz: int | None
    source: str

class SystemInfo(TypedDict):
    cpu_percent: float
    disk_free_gb: float
    disk_total_gb: float
    ram_available_gb: float
    ram_total_gb: float
    ram_percent: float
    gpus: List[GpuInfo]

def _round_gb(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / _BYTES_PER_GB, 2)

@lru_cache(None)
def _which(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def _run_cmd(args: list[str], timeout: float = 8.0) -> str | None:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None

# --- 1. NVIDIA via pynvml -------------------------------------------------

_nvml_initialized = False
_nvml_handles = []

def _init_nvidia():
    global _nvml_initialized, _nvml_handles
    if not _nvml_initialized:
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            _nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            _nvml_initialized = True
        except Exception:
            pass

def _collect_nvidia() -> List[GpuInfo]:
    _init_nvidia()
    try:
        import pynvml
    except ImportError:
        return []

    gpus: List[GpuInfo] = []
    for handle in _nvml_handles:
        try:
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")

            util = None
            try:
                util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            except Exception:
                pass

            vram_used = vram_total = None
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_used = _round_gb(mem.used)
                vram_total = _round_gb(mem.total)
            except Exception:
                pass

            temp = None
            try:
                temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass

            g_clk = None
            try: g_clk = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))
            except: pass
            m_clk = None
            try: m_clk = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
            except: pass

            gpus.append({
                "name": str(name),
                "gpu_util_percent": util,
                "vram_used_gb": vram_used,
                "vram_total_gb": vram_total,
                "temperature_c": temp,
                "graphics_clock_mhz": g_clk,
                "memory_clock_mhz": m_clk,
                "source": "nvidia",
            })
        except Exception as exc:
            logger.debug("NVIDIA collection failed: %s", exc)
    return gpus

# --- 2. Intel Arc via xpu-smi / oneAPI ------------------------------------

def _parse_intel_json(payload: str) -> List[GpuInfo]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []

    devices: list[Any]
    if isinstance(data, list):
        devices = data
    elif isinstance(data, dict):
        for key in ("device_list", "devices", "deviceList", "DeviceList", "gpu_list"):
            value = data.get(key)
            if isinstance(value, list):
                devices = value
                break
        else:
            devices = [data]
    else:
        return []

    gpus: List[GpuInfo] = []
    for device in devices:
        if not isinstance(device, dict):
            continue

        name = (device.get("device_name") or device.get("Device Name") or device.get("name") or 
                 device.get("Name") or device.get("model") or "Intel GPU")

        util = (device.get("gpu_utilization") or device.get("GPU Utilization (%)") or 
                 device.get("compute_util") or device.get("utilization"))
        mem_used = (device.get("gpu_memory_used") or device.get("Memory Used (MiB)") or 
                     device.get("vram_used") or device.get("dedicated_memory_used"))
        mem_total = (device.get("gpu_memory_total") or device.get("Memory Total (MiB)") or 
                      device.get("vram_total") or device.get("dedicated_memory_total"))
        temp = (device.get("gpu_temperature") or device.get("GPU Temperature (C)") or 
                 device.get("temperature") or device.get("temp"))

        def _as_float(value: Any) -> float | None:
            if value is None: return None
            match = re.search(r"[-+]?\d*\.?\d+", str(value))
            return float(match.group(0)) if match else None

        util_f = _as_float(util)
        used_f = _as_float(mem_used)
        total_f = _as_float(mem_total)
        temp_f = _as_float(temp)

        if used_f is not None and used_f > 64:
            used_gb = round(used_f / 1024, 2)
        else:
            used_gb = used_f
        if total_f is not None and total_f > 64:
            total_gb = round(total_f / 1024, 2)
        else:
            total_gb = total_f

        g_clk = None
        try: g_clk = int(_as_float(device.get("graphics_clock") or device.get("Graphics Clock")))
        except: pass
        m_clk = None
        try: m_clk = int(_as_float(device.get("memory_clock") or device.get("Memory Clock")))
        except: pass

        gpus.append({
            "name": str(name),
            "gpu_util_percent": util_f,
            "vram_used_gb": used_gb,
            "vram_total_gb": total_gb,
            "temperature_c": temp_f,
            "graphics_clock_mhz": g_clk,
            "memory_clock_mhz": m_clk,
            "source": "intel",
        })
    return gpus

def _collect_intel() -> List[GpuInfo]:
    if not _which("xpu-smi"):
        return []
    for args in (["xpu-smi", "stats", "-j"], ["xpu-smi", "discovery", "-j"], ["xpu-smi", "dump", "-j", "-d", "0", "-m", "0,18,19"]):
        output = _run_cmd(args)
        if not output: continue
        gpus = _parse_intel_json(output)
        if gpus: return gpus
    return []

# --- 3. AMD via amd-smi / rocm-smi / WMI ----------------------------------

def _collect_amd_smi() -> List[GpuInfo]:
    if not _which("amd-smi"):
        return []
    output = _run_cmd(["amd-smi", "metric", "--json"]) or _run_cmd(["amd-smi", "static", "--json"])
    if not output: return []
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return []
    rows = data if isinstance(data, list) else data.get("gpu_data") or data.get("devices") or [data]
    gpus: List[GpuInfo] = []
    for row in rows:
        if not isinstance(row, dict): continue
        name = (row.get("market_name") or row.get("card_series") or row.get("name") or 
                 ((row.get("asic") or {}).get("market_name") if isinstance(row.get("asic"), dict) else None) or "AMD GPU")
        util = row.get("gfx_activity") or row.get("gpu_activity") or row.get("usage")
        if isinstance(util, dict): util = util.get("gfx") or util.get("value")
        mem = row.get("vram") or row.get("memory") or {}
        used = mem.get("used") or mem.get("vram_used_bytes")
        total = mem.get("total") or mem.get("vram_total_bytes")
        temp = row.get("temperature") or row.get("temp")
        if isinstance(temp, dict): temp = temp.get("hotspot") or temp.get("edge") or temp.get("value")

        def _num(value: Any) -> float | None:
            try: return float(value) if value is not None else None
            except: return None

        u_n, us_n, tot_n, t_n = _num(util), _num(used), _num(total), _num(temp)
        us_gb = _round_gb(us_n) if us_n and us_n > 10000 else (round(us_n / 1024, 2) if us_n and us_n > 64 else us_n) if us_n else None
        tot_gb = _round_gb(tot_n) if tot_n and tot_n > 10000 else (round(tot_n / 1024, 2) if tot_n and tot_n > 64 else tot_n) if tot_n else None

        gpus.append({
            "name": str(name),
            "gpu_util_percent": u_n,
            "vram_used_gb": us_gb,
            "vram_total_gb": tot_gb,
            "temperature_c": t_n,
            "graphics_clock_mhz": None,
            "memory_clock_mhz": None,
            "source": "amd",
        })
    return gpus

def _collect_rocm_smi() -> List[GpuInfo]:
    if not _which("rocm-smi"): return []
    output = _run_cmd(["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp", "--json"])
    if not output: return []
    try:
        data = json.loads(output)
    except json.JSONDecodeError: return []
    gpus: List[GpuInfo] = []
    if not isinstance(data, dict): return []
    for card_id, info in data.items():
        if not isinstance(info, dict): continue
        name = info.get("Card series") or info.get("Device Name") or str(card_id)
        util = info.get("GPU use (%)") or info.get("GPU Utilization (%)")
        used = info.get("VRAM Total Used Memory (B)") or info.get("vram_used")
        total = info.get("VRAM Total Memory (B)") or info.get("vram_total")
        temp = info.get("Temperature (Sensor edge) (C)") or info.get("Temperature (C)")

        def _num(value: Any) -> float | None:
            try: return float(str(value).replace("%", "").strip()) if value is not None else None
            except: return None

        u_n, us_n, tot_n, t_n = _num(util), _num(used), _num(total), _num(temp)
        gpus.append({
            "name": str(name),
            "gpu_util_percent": u_n,
            "vram_used_gb": _round_gb(us_n) if us_n is not None else None,
            "vram_total_gb": _round_gb(tot_n) if tot_n is not None else None,
            "temperature_c": t_n,
            "graphics_clock_mhz": None,
            "memory_clock_mhz": None,
            "source": "amd",
        })
    return gpus

def _collect_amd() -> List[GpuInfo]:
    return _collect_amd_smi() or _collect_rocm_smi() or _collect_amd_wmi()

# --- 4. Windows PDH / Get-Counter fallback --------------------------------

def _wmi_video_controllers() -> List[dict]:
    output = _run_cmd(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, PNPDeviceID | ConvertTo-Json -Compress"])
    if not output: return []
    try:
        rows = json.loads(output)
    except json.JSONDecodeError: return []
    if isinstance(rows, dict): rows = [rows]
    return [r for r in rows if isinstance(r, dict) and (str(r.get("Name")).strip() and str(r.get("Name")).lower() not in {"microsoft basic display adapter"})]

def _pdh_gpu_metrics() -> dict:
    if sys.platform != "win32": return {}
    try:
        import win32pdh
    except ImportError:
        return _powershell_counter_gpu_metrics()
    metrics = {}
    def _ensure(luid): return metrics.setdefault(luid, {"gpu_util_percent": 0.0, "vram_used_bytes": None, "samples": 0.0})
    try:
        u_paths = win32pdh.ExpandWildCardPath(r"\GPU Engine(*)\Utilization Percentage")
        m_paths = win32pdh.ExpandWildCardPath(r"\GPU Adapter Memory(*)\Dedicated Usage")
        if not u_paths and not m_paths: return _powershell_counter_gpu_metrics()
        query = win32pdh.OpenQuery()
        u_counters, m_counters = [], []
        luid_re = re.compile(r"luid_0x([0-9A-Fa-f]+)_0x([0-9A-Fa-f]+)", re.I)
        for path in u_paths or []:
            if "engtype_3D" not in path.lower(): continue
            match = luid_re.search(path)
            if match:
                luid = f"{match.group(1)}_{match.group(2)}".lower()
                u_counters.append((win32pdh.AddCounter(query, path), luid))
        for path in m_paths or []:
            match = luid_re.search(path)
            if match:
                luid = f"{match.group(1)}_{match.group(2)}".lower()
                m_counters.append((win32pdh.AddCounter(query, path), luid))
        win32pdh.CollectQueryData(query)
        time.sleep(0.4)
        win32pdh.CollectQueryData(query)
        for counter, luid in u_counters:
            try:
                _, val = win32pdh.GetFormattedCounterValue(counter, win32pdh.PDH_FMT_DOUBLE)
            except Exception:
                continue
            bucket = _ensure(luid)
            bucket["gpu_util_percent"] += float(val)
            bucket["samples"] += 1.0
        for counter, luid in m_counters:
            try:
                _, val = win32pdh.GetFormattedCounterValue(counter, win32pdh.PDH_FMT_DOUBLE)
            except Exception:
                continue
            bucket = _ensure(luid)
            bucket["vram_used_bytes"] = float(val) + (float(bucket["vram_used_bytes"]) if bucket["vram_used_bytes"] else 0.0)
        win32pdh.CloseQuery(query)

    except Exception: return _powershell_counter_gpu_metrics()
    cleaned = {}
    for luid, b in metrics.items():
        util = max(0.0, min(100.0, float(b["gpu_util_percent"]))) if b["gpu_util_percent"] is not None else None
        cleaned[luid] = {"gpu_util_percent": util, "vram_used_bytes": b["vram_used_bytes"]}
    return cleaned

def _powershell_counter_gpu_metrics() -> dict:
    script = r"""$ErrorActionPreference = 'SilentlyContinue'; $util = Get-Counter '\GPU Engine(*engtype_3D*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 1; $mem  = Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -SampleInterval 1 -MaxSamples 1; $result = @{}; foreach ($sample in $util.CounterSamples) { if ($sample.InstanceName -match 'luid_0x([0-9A-Fa-f]+)_0x([0-9A-Fa-f]+)') { $luid = ($Matches[1] + '_' + $Matches[2]).ToLower(); if (-not $result.ContainsKey($luid)) { $result[$luid] = @{ util = 0.0; vram = $null } }; $result[$luid].util += [double]$sample.CookedValue }; foreach ($sample in $mem.CounterSamples) { if ($sample.InstanceName -match 'luid_0x([0-9A-Fa-f]+)_0x([0-9A-Fa-f]+)') { $luid = ($Matches[1] + '_' + $Matches[2]).ToLower(); if (-not $result.ContainsKey($luid)) { $result[$luid] = @{ util = 0.0; vram = $null } }; $cur = $result[$luid].vram; $val = [double]$sample.CookedValue; $result[$luid].vram = if ($null -eq $cur) { $val } else { $cur + $val } }; $result.GetEnumerator() | ForEach-Object { [pscustomobject]@{ luid = $_.Key; util = [math]::Min(100.0, [double]$_.Value.util); vram = $_.Value.vram } } | ConvertTo-Json -Compress """
    output = _run_cmd(["powershell", "-NoProfile", "-Command", script], timeout=15.0)
    if not output: return {}
    try: rows = json.loads(output)
    except: return {}
    if isinstance(rows, dict): rows = [rows]
    metrics = {}
    for row in rows:
        luid = str(row.get("luid") or "")
        if not luid: continue
        try: metrics[luid] = {"gpu_util_percent": float(row.get("util")), "vram_used_bytes": float(row.get("vram"))}
        except: pass
    return metrics

def _collect_windows() -> List[GpuInfo]:
    if sys.platform != "win32": return []
    controllers = _wmi_video_controllers()
    pdh = _pdh_gpu_metrics()
    pdh_values = list(pdh.values())
    gpus: List[GpuInfo] = []
    if controllers:
        for index, controller in enumerate(controllers):
            name = str(controller.get("Name") or f"GPU {index}")
            adapter_ram = controller.get("AdapterRAM")
            total_gb = _round_gb(int(adapter_ram) & 0xFFFFFFFF) if adapter_ram and int(adapter_ram) > 0 else None
            metrics = pdh_values[index] if index < len(pdh_values) else {}
            used_bytes = metrics.get("vram_used_bytes")
            util = metrics.get("gpu_util_percent")
            gpus.append({
                "name": name,
                "gpu_util_percent": round(util, 1) if isinstance(util, (int, float)) else None,
                "vram_used_gb": _round_gb(used_bytes) if used_bytes is not None else None,
                "vram_total_gb": total_gb,
                "temperature_c": None,
                "graphics_clock_mhz": None,
                "memory_clock_mhz": None,
                "source": "windows",
            })
        return gpus
    for index, metrics in enumerate(pdh_values):
        util = metrics.get("gpu_util_percent")
        used_bytes = metrics.get("vram_used_bytes")
        gpus.append({
            "name": f"GPU {index}",
            "gpu_util_percent": round(util, 1) if isinstance(util, (int, float)) else None,
            "vram_used_gb": _round_gb(used_bytes) if used_bytes is not None else None,
            "vram_total_gb": None,
            "temperature_c": None,
            "graphics_clock_mhz": None,
            "memory_clock_mhz": None,
            "source": "windows",
        })
    return gpus

def _names_overlap(a: str, b: str) -> bool:
    a_l, b_l = a.lower(), b.lower()
    return a_l in b_l or b_l in a_l

def collect_gpus() -> List[GpuInfo]:
    collected: List[GpuInfo] = []
    for collector in (_collect_nvidia, _collect_intel, _collect_amd):
        try: batch = collector()
        except Exception: batch = []
        for gpu in batch:
            if any(_names_overlap(gpu["name"], existing["name"]) for existing in collected): continue
            collected.append(gpu)
    if collected: return collected
    try: return _collect_windows()
    except Exception: return []

@mcp.tool()
def get_system_info() -> SystemInfo:
    disk = shutil.disk_usage(".")
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "disk_free_gb": round(disk.free / _BYTES_PER_GB, 2),
        "disk_total_gb": round(disk.total / _BYTES_PER_GB, 2),
        "ram_available_gb": round(mem.available / _BYTES_PER_GB, 2),
        "ram_total_gb": round(mem.total / _BYTES_PER_GB, 2),
        "ram_percent": mem.percent,
        "gpus": collect_gpus(),
    }

if __name__ == "__main__":
    mcp.run()
