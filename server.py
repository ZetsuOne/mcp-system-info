"""Локальный MCP-сервер: CPU, диск, RAM и GPU (NVIDIA / Intel Arc / AMD / Windows)."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import psutil
from mcp.server.mcpserver import MCPServer

logger = logging.getLogger(__name__)

mcp = MCPServer("system-info")

_BYTES_PER_GB = 1024**3


def _round_gb(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / _BYTES_PER_GB, 2)


def _gpu_entry(
    *,
    name: str,
    gpu_util_percent: float | None = None,
    vram_used_gb: float | None = None,
    vram_total_gb: float | None = None,
    temperature_c: float | None = None,
    source: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "gpu_util_percent": gpu_util_percent,
        "vram_used_gb": vram_used_gb,
        "vram_total_gb": vram_total_gb,
        "temperature_c": temperature_c,
        "source": source,
    }


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


def _which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


# --- 1. NVIDIA via pynvml -------------------------------------------------


def _collect_nvidia() -> list[dict[str, Any]]:
    try:
        import pynvml
    except ImportError:
        return []

    try:
        pynvml.nvmlInit()
    except Exception:
        return []

    gpus: list[dict[str, Any]] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
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
                temp = float(
                    pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                )
            except Exception:
                pass

            gpus.append(
                _gpu_entry(
                    name=str(name),
                    gpu_util_percent=util,
                    vram_used_gb=vram_used,
                    vram_total_gb=vram_total,
                    temperature_c=temp,
                    source="nvidia",
                )
            )
    except Exception as exc:
        logger.debug("NVIDIA collection failed: %s", exc)
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return gpus


# --- 2. Intel Arc via xpu-smi / oneAPI ------------------------------------


def _parse_intel_json(payload: str) -> list[dict[str, Any]]:
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

    gpus: list[dict[str, Any]] = []
    for device in devices:
        if not isinstance(device, dict):
            continue

        name = (
            device.get("device_name")
            or device.get("Device Name")
            or device.get("name")
            or device.get("Name")
            or device.get("model")
            or "Intel GPU"
        )

        util = (
            device.get("gpu_utilization")
            or device.get("GPU Utilization (%)")
            or device.get("compute_util")
            or device.get("utilization")
        )
        mem_used = (
            device.get("gpu_memory_used")
            or device.get("Memory Used (MiB)")
            or device.get("vram_used")
            or device.get("dedicated_memory_used")
        )
        mem_total = (
            device.get("gpu_memory_total")
            or device.get("Memory Total (MiB)")
            or device.get("vram_total")
            or device.get("dedicated_memory_total")
        )
        temp = (
            device.get("gpu_temperature")
            or device.get("GPU Temperature (C)")
            or device.get("temperature")
            or device.get("temp")
        )

        def _as_float(value: Any) -> float | None:
            if value is None:
                return None
            if isinstance(value, str):
                match = re.search(r"[-+]?\d*\.?\d+", value)
                if not match:
                    return None
                value = match.group(0)
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        util_f = _as_float(util)
        used_f = _as_float(mem_used)
        total_f = _as_float(mem_total)
        temp_f = _as_float(temp)

        # xpu-smi часто отдаёт память в MiB
        if used_f is not None and used_f > 64:
            used_gb = round(used_f / 1024, 2)
        else:
            used_gb = used_f
        if total_f is not None and total_f > 64:
            total_gb = round(total_f / 1024, 2)
        else:
            total_gb = total_f

        gpus.append(
            _gpu_entry(
                name=str(name),
                gpu_util_percent=util_f,
                vram_used_gb=used_gb,
                vram_total_gb=total_gb,
                temperature_c=temp_f,
                source="intel",
            )
        )
    return gpus


def _collect_intel() -> list[dict[str, Any]]:
    if not _which("xpu-smi"):
        return []

    for args in (
        ["xpu-smi", "stats", "-j"],
        ["xpu-smi", "discovery", "-j"],
        ["xpu-smi", "dump", "-j", "-d", "0", "-m", "0,18,19"],
    ):
        output = _run_cmd(args)
        if not output:
            continue
        gpus = _parse_intel_json(output)
        if gpus:
            return gpus
    return []


# --- 3. AMD via amd-smi / rocm-smi / WMI ----------------------------------


def _collect_amd_smi() -> list[dict[str, Any]]:
    if not _which("amd-smi"):
        return []

    output = _run_cmd(["amd-smi", "metric", "--json"]) or _run_cmd(
        ["amd-smi", "static", "--json"]
    )
    if not output:
        return []

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return []

    rows = data if isinstance(data, list) else data.get("gpu_data") or data.get("devices") or [data]
    gpus: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = (
            row.get("market_name")
            or row.get("card_series")
            or row.get("name")
            or ((row.get("asic") or {}).get("market_name") if isinstance(row.get("asic"), dict) else None)
            or "AMD GPU"
        )
        util = row.get("gfx_activity") or row.get("gpu_activity") or row.get("usage")
        if isinstance(util, dict):
            util = util.get("gfx") or util.get("value")
        mem = row.get("vram") or row.get("memory") or {}
        if isinstance(mem, dict):
            used = mem.get("used") or mem.get("vram_used_bytes")
            total = mem.get("total") or mem.get("vram_total_bytes")
        else:
            used = total = None
        temp = row.get("temperature") or row.get("temp")
        if isinstance(temp, dict):
            temp = temp.get("hotspot") or temp.get("edge") or temp.get("value")

        def _num(value: Any) -> float | None:
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        used_n, total_n = _num(used), _num(total)
        # amd-smi может отдавать байты или MiB
        if used_n is not None and used_n > 10_000:
            used_gb = _round_gb(used_n)
        elif used_n is not None:
            used_gb = round(used_n / 1024, 2) if used_n > 64 else used_n
        else:
            used_gb = None
        if total_n is not None and total_n > 10_000:
            total_gb = _round_gb(total_n)
        elif total_n is not None:
            total_gb = round(total_n / 1024, 2) if total_n > 64 else total_n
        else:
            total_gb = None

        gpus.append(
            _gpu_entry(
                name=str(name),
                gpu_util_percent=_num(util),
                vram_used_gb=used_gb,
                vram_total_gb=total_gb,
                temperature_c=_num(temp),
                source="amd",
            )
        )
    return gpus


def _collect_rocm_smi() -> list[dict[str, Any]]:
    if not _which("rocm-smi"):
        return []

    output = _run_cmd(
        ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp", "--json"]
    )
    if not output:
        return []

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return []

    gpus: list[dict[str, Any]] = []
    if not isinstance(data, dict):
        return []

    for card_id, info in data.items():
        if not isinstance(info, dict):
            continue
        name = info.get("Card series") or info.get("Device Name") or str(card_id)
        util = info.get("GPU use (%)") or info.get("GPU Utilization (%)")
        used = info.get("VRAM Total Used Memory (B)") or info.get("vram_used")
        total = info.get("VRAM Total Memory (B)") or info.get("vram_total")
        temp = info.get("Temperature (Sensor edge) (C)") or info.get("Temperature (C)")

        def _num(value: Any) -> float | None:
            try:
                return float(str(value).replace("%", "").strip()) if value is not None else None
            except (TypeError, ValueError):
                return None

        gpus.append(
            _gpu_entry(
                name=str(name),
                gpu_util_percent=_num(util),
                vram_used_gb=_round_gb(_num(used)) if _num(used) is not None else None,
                vram_total_gb=_round_gb(_num(total)) if _num(total) is not None else None,
                temperature_c=_num(temp),
                source="amd",
            )
        )
    return gpus


def _collect_amd_wmi() -> list[dict[str, Any]]:
    if sys.platform != "win32":
        return []
    try:
        import wmi  # type: ignore
    except ImportError:
        # Без доп. пакета — через PowerShell CIM
        output = _run_cmd(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController |"
                " Select-Object Name, AdapterRAM |"
                " ConvertTo-Json -Compress",
            ]
        )
        if not output:
            return []
        try:
            rows = json.loads(output)
        except json.JSONDecodeError:
            return []
        if isinstance(rows, dict):
            rows = [rows]
        gpus: list[dict[str, Any]] = []
        for row in rows:
            name = str(row.get("Name") or "")
            if "amd" not in name.lower() and "radeon" not in name.lower():
                continue
            adapter_ram = row.get("AdapterRAM")
            total = None
            try:
                if adapter_ram is not None and int(adapter_ram) > 0:
                    total = _round_gb(int(adapter_ram) & 0xFFFFFFFF)
                    # AdapterRAM часто 32-bit truncated; если явно мало — оставим как есть
            except (TypeError, ValueError):
                total = None
            gpus.append(
                _gpu_entry(
                    name=name,
                    gpu_util_percent=None,
                    vram_used_gb=None,
                    vram_total_gb=total,
                    temperature_c=None,
                    source="amd",
                )
            )
        return gpus

    try:
        client = wmi.WMI()
        gpus = []
        for controller in client.Win32_VideoController():
            name = str(controller.Name or "")
            if "amd" not in name.lower() and "radeon" not in name.lower():
                continue
            total = None
            try:
                if controller.AdapterRAM:
                    total = _round_gb(int(controller.AdapterRAM) & 0xFFFFFFFF)
            except Exception:
                total = None
            gpus.append(
                _gpu_entry(
                    name=name,
                    vram_total_gb=total,
                    temperature_c=None,
                    source="amd",
                )
            )
        return gpus
    except Exception:
        return []


def _collect_amd() -> list[dict[str, Any]]:
    return _collect_amd_smi() or _collect_rocm_smi() or _collect_amd_wmi()


# --- 4. Windows PDH / Get-Counter fallback --------------------------------


def _wmi_video_controllers() -> list[dict[str, Any]]:
    output = _run_cmd(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController |"
            " Select-Object Name, AdapterRAM, PNPDeviceID |"
            " ConvertTo-Json -Compress",
        ]
    )
    if not output:
        return []
    try:
        rows = json.loads(output)
    except json.JSONDecodeError:
        return []
    if isinstance(rows, dict):
        rows = [rows]
    result = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("Name") or "").strip()
        if not name or name.lower() in {"microsoft basic display adapter"}:
            continue
        result.append(row)
    return result


def _pdh_gpu_metrics() -> dict[str, dict[str, float | None]]:
    """Вернуть метрики по LUID: util %, dedicated VRAM bytes."""
    if sys.platform != "win32":
        return {}

    try:
        import win32pdh  # type: ignore
    except ImportError:
        return _powershell_counter_gpu_metrics()

    metrics: dict[str, dict[str, float | None]] = {}

    def _ensure(luid: str) -> dict[str, float | None]:
        return metrics.setdefault(
            luid, {"gpu_util_percent": 0.0, "vram_used_bytes": None, "samples": 0.0}
        )

    try:
        util_paths = win32pdh.ExpandWildCardPath(
            r"\GPU Engine(*)\Utilization Percentage"
        )
    except Exception:
        util_paths = []

    try:
        mem_paths = win32pdh.ExpandWildCardPath(
            r"\GPU Adapter Memory(*)\Dedicated Usage"
        )
    except Exception:
        mem_paths = []

    if not util_paths and not mem_paths:
        return _powershell_counter_gpu_metrics()

    try:
        query = win32pdh.OpenQuery()
        util_counters: list[tuple[Any, str]] = []
        mem_counters: list[tuple[Any, str]] = []

        luid_re = re.compile(r"luid_0x([0-9A-Fa-f]+)_0x([0-9A-Fa-f]+)", re.I)

        for path in util_paths or []:
            # Берём engtype_3D — ближе к «загрузке ядра»
            if "engtype_3D" not in path and "engtype_3d" not in path.lower():
                continue
            match = luid_re.search(path)
            if not match:
                continue
            luid = f"{match.group(1)}_{match.group(2)}".lower()
            try:
                counter = win32pdh.AddCounter(query, path)
                util_counters.append((counter, luid))
            except Exception:
                continue

        for path in mem_paths or []:
            match = luid_re.search(path)
            if not match:
                continue
            luid = f"{match.group(1)}_{match.group(2)}".lower()
            try:
                counter = win32pdh.AddCounter(query, path)
                mem_counters.append((counter, luid))
            except Exception:
                continue

        if not util_counters and not mem_counters:
            win32pdh.CloseQuery(query)
            return _powershell_counter_gpu_metrics()

        win32pdh.CollectQueryData(query)
        time.sleep(0.4)
        win32pdh.CollectQueryData(query)

        for counter, luid in util_counters:
            try:
                _, value = win32pdh.GetFormattedCounterValue(
                    counter, win32pdh.PDH_FMT_DOUBLE
                )
                bucket = _ensure(luid)
                bucket["gpu_util_percent"] = float(bucket["gpu_util_percent"] or 0.0) + float(
                    value
                )
                bucket["samples"] = float(bucket["samples"] or 0.0) + 1.0
            except Exception:
                continue

        for counter, luid in mem_counters:
            try:
                _, value = win32pdh.GetFormattedCounterValue(
                    counter, win32pdh.PDH_FMT_DOUBLE
                )
                bucket = _ensure(luid)
                prev = bucket["vram_used_bytes"]
                bucket["vram_used_bytes"] = float(value) + (
                    float(prev) if prev is not None else 0.0
                )
            except Exception:
                continue

        win32pdh.CloseQuery(query)
    except Exception as exc:
        logger.debug("win32pdh GPU metrics failed: %s", exc)
        return _powershell_counter_gpu_metrics()

    # Нормализуем util (сумма по 3D engine instances → cap 100)
    cleaned: dict[str, dict[str, float | None]] = {}
    for luid, bucket in metrics.items():
        util = bucket.get("gpu_util_percent")
        samples = bucket.get("samples") or 0.0
        if util is not None and samples > 1:
            util = util  # уже сумма по engine instances одного LUID
        if util is not None:
            util = max(0.0, min(100.0, float(util)))
        cleaned[luid] = {
            "gpu_util_percent": util,
            "vram_used_bytes": bucket.get("vram_used_bytes"),
        }
    return cleaned


def _powershell_counter_gpu_metrics() -> dict[str, dict[str, float | None]]:
    script = r"""
$ErrorActionPreference = 'SilentlyContinue'
$util = Get-Counter '\GPU Engine(*engtype_3D*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 1
$mem  = Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -SampleInterval 1 -MaxSamples 1
$result = @{}
foreach ($sample in $util.CounterSamples) {
  if ($sample.InstanceName -match 'luid_0x([0-9A-Fa-f]+)_0x([0-9A-Fa-f]+)') {
    $luid = ($Matches[1] + '_' + $Matches[2]).ToLower()
    if (-not $result.ContainsKey($luid)) { $result[$luid] = @{ util = 0.0; vram = $null } }
    $result[$luid].util += [double]$sample.CookedValue
  }
}
foreach ($sample in $mem.CounterSamples) {
  if ($sample.InstanceName -match 'luid_0x([0-9A-Fa-f]+)_0x([0-9A-Fa-f]+)') {
    $luid = ($Matches[1] + '_' + $Matches[2]).ToLower()
    if (-not $result.ContainsKey($luid)) { $result[$luid] = @{ util = 0.0; vram = $null } }
    $cur = $result[$luid].vram
    $val = [double]$sample.CookedValue
    $result[$luid].vram = if ($null -eq $cur) { $val } else { $cur + $val }
  }
}
$result.GetEnumerator() | ForEach-Object {
  [pscustomobject]@{
    luid = $_.Key
    util = [math]::Min(100.0, [double]$_.Value.util)
    vram = $_.Value.vram
  }
} | ConvertTo-Json -Compress
"""
    output = _run_cmd(["powershell", "-NoProfile", "-Command", script], timeout=15.0)
    if not output:
        return {}
    try:
        rows = json.loads(output)
    except json.JSONDecodeError:
        return {}
    if isinstance(rows, dict):
        rows = [rows]
    metrics: dict[str, dict[str, float | None]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        luid = str(row.get("luid") or "")
        if not luid:
            continue
        util = row.get("util")
        vram = row.get("vram")
        try:
            util_f = float(util) if util is not None else None
        except (TypeError, ValueError):
            util_f = None
        try:
            vram_f = float(vram) if vram is not None else None
        except (TypeError, ValueError):
            vram_f = None
        metrics[luid] = {"gpu_util_percent": util_f, "vram_used_bytes": vram_f}
    return metrics


def _collect_windows() -> list[dict[str, Any]]:
    if sys.platform != "win32":
        return []

    controllers = _wmi_video_controllers()
    pdh = _pdh_gpu_metrics()
    pdh_values = list(pdh.values())

    gpus: list[dict[str, Any]] = []
    if controllers:
        for index, controller in enumerate(controllers):
            name = str(controller.get("Name") or f"GPU {index}")
            adapter_ram = controller.get("AdapterRAM")
            total_gb = None
            try:
                if adapter_ram is not None and int(adapter_ram) > 0:
                    # Win32 AdapterRAM часто обрезан до 4 ГБ — всё равно отдаём как есть
                    total_gb = _round_gb(int(adapter_ram) & 0xFFFFFFFF)
                    if total_gb is not None and total_gb <= 0:
                        total_gb = None
            except (TypeError, ValueError):
                total_gb = None

            metrics = pdh_values[index] if index < len(pdh_values) else {}
            used_bytes = metrics.get("vram_used_bytes") if metrics else None
            util = metrics.get("gpu_util_percent") if metrics else None

            gpus.append(
                _gpu_entry(
                    name=name,
                    gpu_util_percent=round(util, 1) if isinstance(util, (int, float)) else None,
                    vram_used_gb=_round_gb(used_bytes) if used_bytes is not None else None,
                    vram_total_gb=total_gb,
                    temperature_c=None,  # стандартные API Windows температуру GPU не дают
                    source="windows",
                )
            )
        return gpus

    # Нет Win32_VideoController — только PDH
    for index, metrics in enumerate(pdh_values):
        util = metrics.get("gpu_util_percent")
        used_bytes = metrics.get("vram_used_bytes")
        gpus.append(
            _gpu_entry(
                name=f"GPU {index}",
                gpu_util_percent=round(util, 1) if isinstance(util, (int, float)) else None,
                vram_used_gb=_round_gb(used_bytes) if used_bytes is not None else None,
                vram_total_gb=None,
                temperature_c=None,
                source="windows",
            )
        )
    return gpus


# --- Cascade --------------------------------------------------------------


def _names_overlap(a: str, b: str) -> bool:
    a_l, b_l = a.lower(), b.lower()
    return a_l in b_l or b_l in a_l


def collect_gpus() -> list[dict[str, Any]]:
    """Каскад: NVIDIA → Intel Arc → AMD → Windows PDH fallback."""
    collected: list[dict[str, Any]] = []

    for collector in (_collect_nvidia, _collect_intel, _collect_amd):
        try:
            batch = collector()
        except Exception as exc:
            logger.debug("GPU collector failed: %s", exc)
            batch = []
        for gpu in batch:
            if any(_names_overlap(gpu["name"], existing["name"]) for existing in collected):
                continue
            collected.append(gpu)

    if collected:
        return collected

    try:
        return _collect_windows()
    except Exception as exc:
        logger.debug("Windows GPU fallback failed: %s", exc)
        return []


@mcp.tool()
def get_system_info() -> dict:
    """Вернуть загрузку CPU, свободное место на диске, RAM и сведения о GPU."""
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
