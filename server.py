"""Локальный MCP-сервер: CPU, диск, RAM и GPU (NVIDIA / Intel Arc / AMD / Windows)."""

from __future__ import annotations

import base64
import json
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from functools import lru_cache
from typing import Any, Callable, TypedDict

import psutil
from mcp.server.mcpserver import MCPServer

logger = logging.getLogger(__name__)
mcp = MCPServer("system-info")

_BYTES_PER_GB = 1024**3
_CACHE_TTL_SECONDS = 1.5
_CREATIONFLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
_LUID_RE = re.compile(r"luid_0x([0-9A-Fa-f]+)_0x([0-9A-Fa-f]+)", re.I)
_GPU_NAME_DROP = {
    "nvidia", "geforce", "amd", "radeon", "intel", "graphics", "gpu",
    "adapter", "vga", "display", "series", "pci", "pcie", "sse2", "oem",
    "corporation", "corp",
}
_IGNORED_GPU_NAMES = {
    "microsoft basic display adapter",
    "microsoft basic render driver",
}

_nvml_lock = threading.Lock()
_nvml_initialized = False
_nvml_handles: list[Any] = []
_cache_lock = threading.Lock()
_response_cache: dict[bool, tuple[float, dict[str, Any]]] = {}

try:
    psutil.cpu_percent(interval=None)
except Exception:
    logger.debug("initial CPU sample failed", exc_info=True)


class GpuInfo(TypedDict):
    name: str
    gpu_util_percent: float | None
    vram_used_gb: float | None
    vram_total_gb: float | None
    temperature_c: float | None
    graphics_clock_mhz: int | None
    memory_clock_mhz: int | None
    power_watts: float | None
    fan_percent: float | None
    power_limit_watts: float | None
    source: str


class DiskInfo(TypedDict):
    path: str
    free_gb: float
    total_gb: float
    percent: float | None
    fstype: str


class SystemInfo(TypedDict):
    hostname: str
    os: str
    os_release: str
    arch: str
    cpu_model: str | None
    cpu_count: int | None
    cpu_count_physical: int | None
    cpu_percent: float
    uptime_seconds: float | None
    disk_path: str
    disk_free_gb: float
    disk_total_gb: float
    disk_percent: float | None
    disks: list[DiskInfo]
    ram_available_gb: float
    ram_total_gb: float
    ram_percent: float
    gpus: list[GpuInfo]
    warnings: list[str]


def _gpu(**fields: Any) -> GpuInfo:
    gpu: GpuInfo = {
        "name": "GPU",
        "gpu_util_percent": None,
        "vram_used_gb": None,
        "vram_total_gb": None,
        "temperature_c": None,
        "graphics_clock_mhz": None,
        "memory_clock_mhz": None,
        "power_watts": None,
        "fan_percent": None,
        "power_limit_watts": None,
        "source": "windows",
    }
    gpu.update(fields)
    return gpu


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _public_copy(info: dict[str, Any]) -> dict[str, Any]:
    return {
        **info,
        "gpus": [dict(gpu) for gpu in info.get("gpus") or []],
        "disks": [dict(disk) for disk in info.get("disks") or []],
        "warnings": list(info.get("warnings") or []),
    }


def _memo(ttl: float) -> Callable:
    def decorate(fn: Callable) -> Callable:
        box: dict[str, Any] = {"at": 0.0, "key": None, "value": None}
        lock = threading.Lock()

        def wrapper(*args: Any) -> Any:
            now = time.monotonic()
            with lock:
                if box["key"] == args and box["value"] is not None and now - box["at"] < ttl:
                    return box["value"]
            value = fn(*args)
            with lock:
                box["at"] = time.monotonic()
                box["key"] = args
                box["value"] = value
            return value

        return wrapper

    return decorate


def _round_gb(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / _BYTES_PER_GB, 2)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(match.group(0)) if match else None


def _memory_to_gb(value: Any, unit: str | None = None) -> float | None:
    """Переводит объём памяти в ГБ. Единица из имени поля важнее эвристики по величине."""
    number = _as_float(value)
    if number is None:
        return None
    hint = (unit or "").lower()
    if "byte" in hint or hint == "b":
        return round(number / _BYTES_PER_GB, 2)
    if "mib" in hint or hint == "mb":
        return round(number / 1024, 2)
    if "gib" in hint or hint == "gb":
        return round(number, 2)
    if abs(number) >= 1_000_000:
        return round(number / _BYTES_PER_GB, 2)
    if abs(number) > 64:
        return round(number / 1024, 2)
    return round(number, 2)


def _clock(value: Any) -> int | None:
    number = _as_float(value)
    if number is None:
        return None
    return int(number)


def _first_value(mapping: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _pick(device: dict, keys: tuple[tuple[str, str | None], ...]) -> tuple[Any, str | None]:
    for key, unit in keys:
        if key in device and device[key] is not None:
            return device[key], unit
    return None, None


def _mem_gb_from(container: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in container or container[key] is None:
            continue
        value = container[key]
        hint = "bytes" if str(key).endswith("bytes") or str(key).endswith("(B)") else None
        if isinstance(value, dict):
            hint = str(value.get("unit") or hint or "")
            value = value.get("value")
        return _memory_to_gb(value, hint)
    return None


@lru_cache(maxsize=None)
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
            creationflags=_CREATIONFLAGS,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _run_powershell(script: str, timeout: float) -> str | None:
    if sys.platform != "win32":
        return None
    wrapped = (
        "$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n"
        + script
    )
    encoded = base64.b64encode(wrapped.encode("utf-16le")).decode("ascii")
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace",
            creationflags=_CREATIONFLAGS,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        logger.debug("powershell failed: %s", (completed.stderr or "").strip())
        return None
    return completed.stdout.strip() or None


def _gpu_name_tokens(name: str) -> tuple[str, ...]:
    text = name.lower().replace("(r)", " ").replace("(tm)", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return tuple(token for token in text.split() if token not in _GPU_NAME_DROP)


def _same_gpu(left_name: str, right_name: str) -> bool:
    """Одна и та же карта после отбрасывания вендора. 3060 и 3060 Ti остаются разными."""
    left, right = _gpu_name_tokens(left_name), _gpu_name_tokens(right_name)
    return bool(left) and set(left) == set(right)


def _merge_gpu(collected: list[dict[str, Any]], gpu: dict[str, Any]) -> None:
    name = str(gpu.get("name") or "")
    if any(_same_gpu(name, str(item.get("name") or "")) for item in collected):
        return
    collected.append(gpu)


def _ignored_gpu_name(name: str) -> bool:
    lowered = name.lower().strip()
    return lowered in _IGNORED_GPU_NAMES or "microsoft remote display" in lowered


def _luid_key_from_text(text: str) -> str | None:
    match = _LUID_RE.search(text)
    if not match:
        return None
    return f"{int(match.group(1), 16):x}_{int(match.group(2), 16):x}"


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def _adapter_ram_bytes(value: Any) -> int | None:
    """Win32 AdapterRAM — uint32 и не вмещает карты больше 4 ГБ."""
    try:
        if value is None or value == "":
            return None
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        number &= 0xFFFFFFFF
    if number <= 0 or number >= 0xFFFFFFFF:
        return None
    return number


# --- 1. NVIDIA via pynvml -------------------------------------------------

def _init_nvidia(force: bool = False) -> None:
    global _nvml_initialized, _nvml_handles
    with _nvml_lock:
        if _nvml_initialized and not force:
            return
        _nvml_handles = []
        _nvml_initialized = False
        try:
            import pynvml

            if force:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    logger.debug("NVML shutdown before reinit failed", exc_info=True)
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            _nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(count)]
            _nvml_initialized = True
        except Exception:
            logger.debug("NVML init failed", exc_info=True)
            _nvml_handles = []
            _nvml_initialized = False


def _nvml_optional_float(fn: Callable[[], Any], scale: float = 1.0) -> float | None:
    try:
        value = float(fn())
    except Exception:
        return None
    return round(value / scale, 1) if scale != 1.0 else value


def _read_nvidia_devices() -> list[dict[str, Any]]:
    try:
        import pynvml
    except ImportError:
        return []

    gpus: list[dict[str, Any]] = []
    for handle in _nvml_handles:
        try:
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            util = None
            try:
                util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            except Exception:
                logger.debug("NVML utilization unavailable", exc_info=True)
            vram_used = vram_total = None
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_used = _round_gb(mem.used)
                vram_total = _round_gb(mem.total)
            except Exception:
                logger.debug("NVML memory unavailable", exc_info=True)
            temp = None
            try:
                temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                logger.debug("NVML temperature unavailable", exc_info=True)
            graphics_clock = _clock(_nvml_clock(pynvml, handle, "NVML_CLOCK_GRAPHICS"))
            memory_clock = _clock(_nvml_clock(pynvml, handle, "NVML_CLOCK_MEM"))
            power = _nvml_optional_float(lambda: pynvml.nvmlDeviceGetPowerUsage(handle), 1000.0)
            fan = _nvml_optional_float(lambda: pynvml.nvmlDeviceGetFanSpeed(handle))
            limit = _nvml_optional_float(lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(handle), 1000.0)
            if limit is None:
                limit = _nvml_optional_float(lambda: pynvml.nvmlDeviceGetPowerManagementLimit(handle), 1000.0)
            gpus.append(_gpu(
                name=str(name),
                gpu_util_percent=util,
                vram_used_gb=vram_used,
                vram_total_gb=vram_total,
                temperature_c=temp,
                graphics_clock_mhz=graphics_clock,
                memory_clock_mhz=memory_clock,
                power_watts=power,
                fan_percent=fan,
                power_limit_watts=limit,
                source="nvidia",
            ))
        except Exception:
            logger.debug("NVIDIA device read failed", exc_info=True)
    return gpus


def _nvml_clock(pynvml: Any, handle: Any, constant: str) -> int | None:
    try:
        return int(pynvml.nvmlDeviceGetClockInfo(handle, getattr(pynvml, constant)))
    except Exception:
        return None


def _collect_nvidia(warnings: list[str]) -> list[dict[str, Any]]:
    _init_nvidia()
    gpus = _read_nvidia_devices()
    if gpus or not _nvml_handles:
        return gpus
    _init_nvidia(force=True)
    gpus = _read_nvidia_devices()
    if _nvml_handles and not gpus:
        warnings.append("NVIDIA adapters were found but metrics could not be read")
    return gpus


# --- 2. Intel Arc via xpu-smi / oneAPI ------------------------------------

_INTEL_NAME_KEYS = ("device_name", "Device Name", "name", "Name", "model")
_INTEL_UTIL = (
    ("gpu_utilization", None),
    ("GPU Utilization (%)", None),
    ("compute_util", None),
    ("utilization", None),
)
_INTEL_USED = (
    ("gpu_memory_used", None),
    ("Memory Used (MiB)", "mib"),
    ("vram_used", None),
    ("dedicated_memory_used", "bytes"),
)
_INTEL_TOTAL = (
    ("gpu_memory_total", None),
    ("Memory Total (MiB)", "mib"),
    ("vram_total", None),
    ("dedicated_memory_total", "bytes"),
)
_INTEL_TEMP = (
    ("gpu_temperature", None),
    ("GPU Temperature (C)", None),
    ("temperature", None),
    ("temp", None),
)
_INTEL_GCLK = (("graphics_clock", None), ("Graphics Clock", None))
_INTEL_MCLK = (("memory_clock", None), ("Memory Clock", None))


def _parse_intel_json(payload: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []

    if isinstance(data, list):
        devices = data
    elif isinstance(data, dict):
        devices = None
        for key in ("device_list", "devices", "deviceList", "DeviceList", "gpu_list"):
            value = data.get(key)
            if isinstance(value, list):
                devices = value
                break
        if devices is None:
            devices = [data]
    else:
        return []

    gpus: list[dict[str, Any]] = []
    for device in devices:
        if not isinstance(device, dict):
            continue
        name = "Intel GPU"
        for key in _INTEL_NAME_KEYS:
            if device.get(key):
                name = str(device[key])
                break
        util, _ = _pick(device, _INTEL_UTIL)
        used, used_unit = _pick(device, _INTEL_USED)
        total, total_unit = _pick(device, _INTEL_TOTAL)
        temp, _ = _pick(device, _INTEL_TEMP)
        graphics, _ = _pick(device, _INTEL_GCLK)
        memory, _ = _pick(device, _INTEL_MCLK)
        gpus.append(_gpu(
            name=name,
            gpu_util_percent=_as_float(util),
            vram_used_gb=_memory_to_gb(used, used_unit),
            vram_total_gb=_memory_to_gb(total, total_unit),
            temperature_c=_as_float(temp),
            graphics_clock_mhz=_clock(graphics),
            memory_clock_mhz=_clock(memory),
            source="intel",
        ))
    return gpus


def _collect_intel(warnings: list[str]) -> list[dict[str, Any]]:
    if not _which("xpu-smi"):
        return []
    attempts = (
        ["xpu-smi", "stats", "-j"],
        ["xpu-smi", "discovery", "-j"],
        ["xpu-smi", "dump", "-j", "-d", "0", "-m", "0,18,19"],
    )
    for args in attempts:
        output = _run_cmd(args)
        if not output:
            continue
        gpus = _parse_intel_json(output)
        if gpus:
            return gpus
    warnings.append("xpu-smi is installed but returned no Intel GPU metrics")
    return []


# --- 3. AMD via amd-smi / rocm-smi / WMI ----------------------------------

def _parse_amd_smi(payload: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("gpu_data") or data.get("devices") or [data]
    else:
        return []

    gpus: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        asic = row.get("asic") if isinstance(row.get("asic"), dict) else {}
        name = (
            row.get("market_name")
            or row.get("card_series")
            or row.get("name")
            or asic.get("market_name")
            or "AMD GPU"
        )
        util = _first_value(row, ("gfx_activity", "gpu_activity", "usage"))
        if isinstance(util, dict):
            util = util.get("gfx") if util.get("gfx") is not None else util.get("value")
        memory = row.get("vram") if isinstance(row.get("vram"), dict) else None
        if memory is None and isinstance(row.get("memory"), dict):
            memory = row["memory"]
        memory = memory or {}
        temp = _first_value(row, ("temperature", "temp"))
        if isinstance(temp, dict):
            temp = _first_value(temp, ("hotspot", "edge", "value"))
        gpus.append(_gpu(
            name=str(name),
            gpu_util_percent=_as_float(util),
            vram_used_gb=_mem_gb_from(memory, ("used", "vram_used_bytes")),
            vram_total_gb=_mem_gb_from(memory, ("total", "vram_total_bytes")),
            temperature_c=_as_float(temp),
            graphics_clock_mhz=_clock(_first_value(row, ("gfx_clock", "graphics_clock"))),
            memory_clock_mhz=_clock(_first_value(row, ("mem_clock", "memory_clock"))),
            source="amd",
        ))
    return gpus


def _parse_rocm_smi(payload: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    gpus: list[dict[str, Any]] = []
    for card_id, info in data.items():
        if not isinstance(info, dict):
            continue
        name = info.get("Card series") or info.get("Device Name") or str(card_id)
        if info.get("VRAM Total Used Memory (B)") is not None:
            used_gb = _memory_to_gb(info.get("VRAM Total Used Memory (B)"), "bytes")
        else:
            used_gb = _memory_to_gb(info.get("vram_used"))
        if info.get("VRAM Total Memory (B)") is not None:
            total_gb = _memory_to_gb(info.get("VRAM Total Memory (B)"), "bytes")
        else:
            total_gb = _memory_to_gb(info.get("vram_total"))
        util = _first_value(info, ("GPU use (%)", "GPU Utilization (%)"))
        temp = _first_value(info, ("Temperature (Sensor edge) (C)", "Temperature (C)"))
        gpus.append(_gpu(
            name=str(name),
            gpu_util_percent=_as_float(util),
            vram_used_gb=used_gb,
            vram_total_gb=total_gb,
            temperature_c=_as_float(temp),
            source="amd",
        ))
    return gpus


def _collect_amd_smi() -> list[dict[str, Any]]:
    if not _which("amd-smi"):
        return []
    output = _run_cmd(["amd-smi", "metric", "--json"]) or _run_cmd(["amd-smi", "static", "--json"])
    if not output:
        return []
    return _parse_amd_smi(output)


def _collect_rocm_smi() -> list[dict[str, Any]]:
    if not _which("rocm-smi"):
        return []
    output = _run_cmd(["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp", "--json"])
    if not output:
        return []
    return _parse_rocm_smi(output)


def _is_amd_name(name: str) -> bool:
    lowered = name.lower()
    return "amd" in lowered or "radeon" in lowered


def _collect_amd_wmi(fast: bool, warnings: list[str]) -> list[dict[str, Any]]:
    if sys.platform != "win32":
        return []
    inventory = _windows_inventory()
    controllers = [row for row in inventory["controllers"] if _is_amd_name(str(row.get("name") or ""))]
    if not controllers:
        return []
    gpus, join_warnings = _join_windows_adapters(
        controllers,
        inventory["luids"],
        inventory["vram"],
        _pdh_gpu_metrics(fast),
    )
    warnings.extend(join_warnings)
    for gpu in gpus:
        gpu["source"] = "amd"
    return gpus


def _collect_amd(fast: bool, warnings: list[str]) -> list[dict[str, Any]]:
    return _collect_amd_smi() or _collect_rocm_smi() or _collect_amd_wmi(fast, warnings)


# --- 4. Windows PDH / registry / WMI --------------------------------------

_INVENTORY_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'

function Convert-ToUInt64([object]$Value) {
  if ($null -eq $Value) { return $null }
  if ($Value -is [byte[]]) {
    $n = [uint64]0
    $len = [Math]::Min(8, $Value.Length)
    for ($i = 0; $i -lt $len; $i++) {
      $n = $n -bor ([uint64]$Value[$i] -shl (8 * $i))
    }
    return $n
  }
  try { return [uint64]$Value } catch { return $null }
}

$controllers = @(
  Get-CimInstance Win32_VideoController | ForEach-Object {
    [pscustomobject]@{
      Name = $_.Name
      AdapterRAM = $_.AdapterRAM
      PNPDeviceID = $_.PNPDeviceID
    }
  }
)

$directx = @(
  Get-ChildItem 'HKLM:\SOFTWARE\Microsoft\DirectX' -ErrorAction SilentlyContinue | ForEach-Object {
    $p = Get-ItemProperty $_.PSPath
    if (-not $p.Description -or $null -eq $p.AdapterLuid) { return }
    $luid = Convert-ToUInt64 $p.AdapterLuid
    if ($null -eq $luid) { return }
    [pscustomobject]@{
      name = [string]$p.Description
      hi = [uint64](($luid -shr 32) -band [uint64]4294967295)
      lo = [uint64]($luid -band [uint64]4294967295)
    }
  }
)

$class = @(
  Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}' -ErrorAction SilentlyContinue | ForEach-Object {
    $p = Get-ItemProperty $_.PSPath
    if (-not $p.DriverDesc) { return }
    [pscustomobject]@{
      name = [string]$p.DriverDesc
      vram_bytes = Convert-ToUInt64 $p.'HardwareInformation.qwMemorySize'
    }
  }
)

[pscustomobject]@{
  controllers = @($controllers)
  directx = @($directx)
  class_vram = @($class)
} | ConvertTo-Json -Compress -Depth 6
"""

_GPU_COUNTER_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
$paths = @(
  '\GPU Engine(*engtype_3D*)\Utilization Percentage',
  '\GPU Adapter Memory(*)\Dedicated Usage',
  '\GPU Adapter Memory(*)\Dedicated Limit'
)
$samples = Get-Counter -Counter $paths -SampleInterval 1 -MaxSamples 1
$result = @{}

function Get-LuidKey([string]$InstanceName) {
  if ($InstanceName -match 'luid_0x([0-9A-Fa-f]+)_0x([0-9A-Fa-f]+)') {
    $hi = [uint64]::Parse($Matches[1], [System.Globalization.NumberStyles]::AllowHexSpecifier)
    $lo = [uint64]::Parse($Matches[2], [System.Globalization.NumberStyles]::AllowHexSpecifier)
    return ('{0:x}_{1:x}' -f $hi, $lo)
  }
  return $null
}

function Ensure-Bucket([string]$Luid) {
  if (-not $result.ContainsKey($Luid)) {
    $result[$Luid] = @{ util = $null; vram = $null; limit = $null }
  }
}

if ($samples) {
  foreach ($sample in @($samples.CounterSamples)) {
    $luid = Get-LuidKey ([string]$sample.InstanceName)
    if (-not $luid) { continue }
    Ensure-Bucket $luid
    $val = [double]$sample.CookedValue
    $path = [string]$sample.Path
    if ($path -like '*Utilization Percentage*') {
      $cur = $result[$luid].util
      if ($null -eq $cur -or $val -gt [double]$cur) { $result[$luid].util = $val }
    } elseif ($path -like '*Dedicated Limit*') {
      $cur = $result[$luid].limit
      if ($null -eq $cur -or $val -gt [double]$cur) { $result[$luid].limit = $val }
    } elseif ($path -like '*Dedicated Usage*') {
      $cur = $result[$luid].vram
      $result[$luid].vram = if ($null -eq $cur) { $val } else { [double]$cur + $val }
    }
  }
}

$result.GetEnumerator() | ForEach-Object {
  $util = $_.Value.util
  if ($null -ne $util) {
    $util = [math]::Min(100.0, [math]::Max(0.0, [double]$util))
  }
  [pscustomobject]@{
    luid = $_.Key
    util = $util
    vram = $_.Value.vram
    limit = $_.Value.limit
  }
} | ConvertTo-Json -Compress
"""


def _parse_windows_inventory(payload: str) -> dict[str, list]:
    empty = {"controllers": [], "luids": [], "vram": []}
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return empty
    if not isinstance(data, dict):
        return empty

    controllers = []
    for row in _as_list(data.get("controllers")):
        if not isinstance(row, dict):
            continue
        name = str(row.get("Name") or row.get("name") or "").strip()
        if not name or _ignored_gpu_name(name):
            continue
        controllers.append({
            "name": name,
            "adapter_ram": row.get("AdapterRAM", row.get("adapter_ram")),
        })

    luids = []
    for row in _as_list(data.get("directx")):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        try:
            luid = f"{int(row.get('hi')):x}_{int(row.get('lo')):x}"
        except (TypeError, ValueError):
            continue
        if name and not _ignored_gpu_name(name):
            luids.append({"name": name, "luid": luid})

    vram = []
    for row in _as_list(data.get("class_vram")):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        try:
            amount = int(row.get("vram_bytes"))
        except (TypeError, ValueError):
            continue
        if name and amount > 0 and not _ignored_gpu_name(name):
            vram.append({"name": name, "bytes": amount})
    return {"controllers": controllers, "luids": luids, "vram": vram}


@_memo(_CACHE_TTL_SECONDS)
def _windows_inventory() -> dict[str, list]:
    if sys.platform != "win32":
        return {"controllers": [], "luids": [], "vram": []}
    output = _run_powershell(_INVENTORY_SCRIPT, timeout=12)
    if not output:
        return {"controllers": [], "luids": [], "vram": []}
    return _parse_windows_inventory(output)


def _empty_gpu_metrics() -> dict[str, Any]:
    return {"gpu_util_percent": None, "vram_used_bytes": None, "vram_total_bytes": None}


def _pdh_query(win32pdh: Any, fast: bool) -> dict[str, dict[str, Any]]:
    util_paths = win32pdh.ExpandWildCardPath(r"\GPU Engine(*)\Utilization Percentage")
    used_paths = win32pdh.ExpandWildCardPath(r"\GPU Adapter Memory(*)\Dedicated Usage")
    limit_paths = win32pdh.ExpandWildCardPath(r"\GPU Adapter Memory(*)\Dedicated Limit")
    query = win32pdh.OpenQuery()
    try:
        counters: list[tuple[Any, str, str]] = []
        if not fast:
            for path in util_paths or []:
                if "engtype_3d" not in path.lower():
                    continue
                luid = _luid_key_from_text(path)
                if luid:
                    counters.append((win32pdh.AddCounter(query, path), luid, "util"))
        for path in used_paths or []:
            luid = _luid_key_from_text(path)
            if luid:
                counters.append((win32pdh.AddCounter(query, path), luid, "used"))
        for path in limit_paths or []:
            luid = _luid_key_from_text(path)
            if luid:
                counters.append((win32pdh.AddCounter(query, path), luid, "limit"))
        if not counters:
            return {}
        win32pdh.CollectQueryData(query)
        if not fast:
            time.sleep(0.4)
            win32pdh.CollectQueryData(query)
        metrics: dict[str, dict[str, Any]] = {}
        for counter, luid, kind in counters:
            try:
                _, raw = win32pdh.GetFormattedCounterValue(counter, win32pdh.PDH_FMT_DOUBLE)
                value = float(raw)
            except Exception:
                continue
            slot = metrics.setdefault(luid, _empty_gpu_metrics())
            if kind == "util":
                current = slot["gpu_util_percent"]
                slot["gpu_util_percent"] = value if current is None else max(float(current), value)
            elif kind == "used":
                current = slot["vram_used_bytes"]
                slot["vram_used_bytes"] = value if current is None else float(current) + value
            else:
                current = slot["vram_total_bytes"]
                slot["vram_total_bytes"] = value if current is None else max(float(current), value)
        for slot in metrics.values():
            util = slot["gpu_util_percent"]
            if isinstance(util, (int, float)):
                slot["gpu_util_percent"] = max(0.0, min(100.0, float(util)))
        return metrics
    finally:
        try:
            win32pdh.CloseQuery(query)
        except Exception:
            logger.debug("PDH CloseQuery failed", exc_info=True)


def _metrics_from_rows(rows: list[Any]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        luid = str(row.get("luid") or "")
        if not luid:
            continue
        slot = _empty_gpu_metrics()
        try:
            if row.get("util") is not None:
                slot["gpu_util_percent"] = max(0.0, min(100.0, float(row["util"])))
        except (TypeError, ValueError):
            pass
        try:
            if row.get("vram") is not None:
                slot["vram_used_bytes"] = float(row["vram"])
        except (TypeError, ValueError):
            pass
        try:
            if row.get("limit") is not None:
                slot["vram_total_bytes"] = float(row["limit"])
        except (TypeError, ValueError):
            pass
        metrics[luid] = slot
    return metrics


def _powershell_counter_gpu_metrics() -> dict[str, dict[str, Any]]:
    output = _run_powershell(_GPU_COUNTER_SCRIPT, timeout=15)
    if not output:
        return {}
    try:
        rows = json.loads(output)
    except json.JSONDecodeError:
        return {}
    return _metrics_from_rows(_as_list(rows))


@_memo(_CACHE_TTL_SECONDS)
def _pdh_gpu_metrics(fast: bool) -> dict[str, dict[str, Any]]:
    if sys.platform != "win32":
        return {}
    try:
        import win32pdh
    except ImportError:
        win32pdh = None
    if win32pdh is not None:
        try:
            return _pdh_query(win32pdh, fast)
        except Exception:
            logger.debug("PDH query failed", exc_info=True)
    if fast:
        return {}
    return _powershell_counter_gpu_metrics()


def _vram_bytes_for(name: str, vram_by_name: list[dict]) -> int | None:
    for row in vram_by_name:
        if _same_gpu(name, str(row.get("name") or "")):
            value = row.get("bytes")
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
    return None


def _luid_for(name: str, luid_names: list[dict]) -> str | None:
    for row in luid_names:
        if _same_gpu(name, str(row.get("name") or "")):
            luid = str(row.get("luid") or "")
            if luid:
                return luid
    return None


def _join_windows_adapters(
    controllers: list[dict],
    luid_names: list[dict],
    vram_by_name: list[dict],
    pdh: dict[str, dict],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Склеивает имя адаптера со счётчиками по LUID, а не по позиции в списке."""
    warnings: list[str] = []
    usable = [row for row in controllers if not _ignored_gpu_name(str(row.get("name") or ""))]
    pdh_items = list(pdh.items())

    def build(name: str, metrics: dict, adapter_ram: Any) -> dict[str, Any]:
        total = metrics.get("vram_total_bytes")
        total_source = "counter" if isinstance(total, (int, float)) and total > 0 else None
        if total_source is None:
            registry_bytes = _vram_bytes_for(name, vram_by_name)
            if registry_bytes:
                total = registry_bytes
                total_source = "registry"
        if total_source is None:
            wmi_bytes = _adapter_ram_bytes(adapter_ram)
            if wmi_bytes:
                total = wmi_bytes
                warnings.append(
                    f"VRAM size for {name} came from Win32_VideoController.AdapterRAM, which is unreliable above 4 GB"
                )
        used = metrics.get("vram_used_bytes")
        util = metrics.get("gpu_util_percent")
        return _gpu(
            name=name,
            gpu_util_percent=round(float(util), 1) if isinstance(util, (int, float)) else None,
            vram_used_gb=_round_gb(used) if isinstance(used, (int, float)) else None,
            vram_total_gb=_round_gb(total) if isinstance(total, (int, float)) and total > 0 else None,
            source="windows",
        )

    if not usable:
        if not pdh_items:
            warnings.append("Windows GPU inventory returned no adapters")
            return [], warnings
        warnings.append("GPU names were not available; counters are reported without adapter names")
        return [build(f"GPU {index}", metrics, None) for index, (_, metrics) in enumerate(pdh_items)], warnings

    matched: list[tuple[dict, str | None]] = []
    match_count = 0
    for controller in usable:
        luid = _luid_for(str(controller.get("name") or ""), luid_names)
        if luid is not None:
            match_count += 1
        matched.append((controller, luid))

    use_index = False
    if match_count == 0 and pdh_items and len(pdh_items) == len(usable):
        use_index = True
        warnings.append("GPU counters could not be matched to adapter names; utilization is aligned by index")
    elif match_count < len(usable) and pdh_items:
        warnings.append("Some GPUs had no LUID match; their utilization was left empty instead of guessing")

    gpus = []
    for index, (controller, luid) in enumerate(matched):
        name = str(controller.get("name") or f"GPU {index}")
        if use_index:
            metrics = pdh_items[index][1]
        elif luid is not None and luid in pdh:
            metrics = pdh[luid]
        else:
            metrics = {}
        gpus.append(build(name, metrics, controller.get("adapter_ram")))
    return gpus, _unique(warnings)


def collect_gpus(fast: bool = False, warnings: list[str] | None = None) -> list[dict[str, Any]]:
    sink = warnings if warnings is not None else []
    collected: list[dict[str, Any]] = []
    steps = (
        ("NVIDIA", lambda: _collect_nvidia(sink)),
        ("Intel", lambda: _collect_intel(sink)),
        ("AMD", lambda: _collect_amd(fast, sink)),
    )
    for label, collector in steps:
        try:
            batch = collector()
        except Exception as exc:
            logger.debug("%s collection failed", label, exc_info=True)
            sink.append(f"{label} collector failed: {exc}")
            batch = []
        for gpu in batch:
            _merge_gpu(collected, gpu)
    if sys.platform == "win32":
        try:
            inventory = _windows_inventory()
            pending = [
                row for row in inventory["controllers"]
                if not any(
                    _same_gpu(str(row.get("name") or ""), str(item.get("name") or ""))
                    for item in collected
                )
            ]
            if pending or not inventory["controllers"]:
                extra, join_warnings = _join_windows_adapters(
                    pending if pending else inventory["controllers"],
                    inventory["luids"],
                    inventory["vram"],
                    _pdh_gpu_metrics(fast),
                )
                sink.extend(join_warnings)
                for gpu in extra:
                    _merge_gpu(collected, gpu)
        except Exception as exc:
            logger.debug("Windows collection failed", exc_info=True)
            sink.append(f"Windows collector failed: {exc}")
    return collected


# --- Host snapshot --------------------------------------------------------

def _disk_record(path: str, fstype: str = "") -> dict[str, Any] | None:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    percent = round((usage.total - usage.free) / usage.total * 100, 1) if usage.total else None
    return {
        "path": path,
        "free_gb": round(usage.free / _BYTES_PER_GB, 2),
        "total_gb": round(usage.total / _BYTES_PER_GB, 2),
        "percent": percent,
        "fstype": fstype,
    }


def _system_disk_path() -> str:
    if sys.platform == "win32":
        return os.environ.get("SystemDrive", "C:") + "\\"
    return "/"


def _collect_disks(warnings: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    disks: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        partitions = psutil.disk_partitions(all=False)
    except Exception as exc:
        warnings.append(f"disk enumeration failed: {exc}")
        partitions = []
    skip_fs = {"squashfs", "overlay", "tmpfs", "devtmpfs", "autofs", "proc", "sysfs", "devfs"}
    for part in partitions:
        opts = (part.opts or "").lower()
        if "cdrom" in opts:
            continue
        if sys.platform != "win32" and part.fstype in skip_fs:
            continue
        mount = part.mountpoint
        if mount in seen:
            continue
        record = _disk_record(mount, part.fstype or "")
        if record is None:
            continue
        seen.add(mount)
        disks.append(record)

    system_path = _system_disk_path()
    primary = next(
        (disk for disk in disks if os.path.normcase(disk["path"]) == os.path.normcase(system_path)),
        None,
    )
    if primary is None:
        primary = _disk_record(system_path)
        if primary is not None:
            disks.insert(0, primary)
        else:
            warnings.append(
                f"system disk {system_path} is not readable; using the process working directory"
            )
            primary = _disk_record(os.getcwd()) or {
                "path": os.getcwd(),
                "free_gb": 0.0,
                "total_gb": 0.0,
                "percent": None,
                "fstype": "",
            }
    return primary, disks


@lru_cache(maxsize=1)
def _cpu_model() -> str | None:
    try:
        if sys.platform == "win32":
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                text = str(value).strip()
                if text:
                    return text
        if os.path.isfile("/proc/cpuinfo"):
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        text = line.split(":", 1)[1].strip()
                        if text:
                            return text
    except Exception:
        logger.debug("cpu model lookup failed", exc_info=True)
    text = platform.processor().strip()
    return text or None


def _gather_system_info(fast: bool) -> SystemInfo:
    warnings: list[str] = []
    primary, disks = _collect_disks(warnings)
    memory = psutil.virtual_memory()
    try:
        cpu = float(psutil.cpu_percent(interval=None if fast else 0.5))
    except Exception as exc:
        warnings.append(f"CPU utilization unavailable: {exc}")
        cpu = 0.0
    try:
        uptime = round(time.time() - psutil.boot_time(), 1)
    except Exception:
        uptime = None
        warnings.append("uptime is unavailable")
    return {
        "hostname": socket.gethostname(),
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_percent": cpu,
        "uptime_seconds": uptime,
        "disk_path": primary["path"],
        "disk_free_gb": primary["free_gb"],
        "disk_total_gb": primary["total_gb"],
        "disk_percent": primary["percent"],
        "disks": disks,
        "ram_available_gb": round(memory.available / _BYTES_PER_GB, 2),
        "ram_total_gb": round(memory.total / _BYTES_PER_GB, 2),
        "ram_percent": memory.percent,
        "gpus": collect_gpus(fast=fast, warnings=warnings),
        "warnings": _unique(warnings),
    }


@mcp.tool()
def get_system_info(fast: bool = False) -> SystemInfo:
    """Snapshot of this machine: CPU, RAM, disks, and GPUs.

    Args:
        fast: Skip the blocking CPU sample and the second GPU-counter sample.
            Utilization may be null or stale. Memory sizes stay accurate.

    GPU cascade: NVIDIA NVML, then Intel xpu-smi, then AMD amd-smi / rocm-smi / WMI.
    Windows performance counters fill in adapters the vendor tools did not report.
    Duplicate names are removed without merging distinct models (3060 and 3060 Ti stay separate).
    `warnings` lists collection problems. Missing sensors are null.
    `disk_*` describes the system volume; `disks` lists mounted fixed volumes.
    """
    now = time.monotonic()
    with _cache_lock:
        cached = _response_cache.get(fast)
        if cached and now - cached[0] < _CACHE_TTL_SECONDS:
            return _public_copy(cached[1])
    snapshot = _gather_system_info(fast)
    with _cache_lock:
        _response_cache[fast] = (time.monotonic(), snapshot)
    return _public_copy(snapshot)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
