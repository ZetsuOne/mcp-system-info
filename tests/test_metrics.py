"""Парсеры и склейка GPU без живого железа."""

from __future__ import annotations

import json

import server


def test_memory_units_follow_the_field_not_just_magnitude() -> None:
    assert server._memory_to_gb(0, "bytes") == 0.0
    assert server._memory_to_gb(96, "gb") == 96.0
    assert server._memory_to_gb(8192, "mib") == 8.0
    assert server._memory_to_gb(25769803776, "bytes") == 24.0
    assert server._memory_to_gb(None) is None


def test_adapter_ram_rejects_the_32_bit_overflow() -> None:
    assert server._adapter_ram_bytes(0xFFFFFFFF) is None
    assert server._adapter_ram_bytes(0) is None
    assert server._adapter_ram_bytes(2 * 1024**3) == 2 * 1024**3
    assert server._adapter_ram_bytes(-1073741824) == 3221225472


def test_same_gpu_keeps_ti_and_xt_variants() -> None:
    assert server._same_gpu("NVIDIA GeForce RTX 3060", "RTX 3060")
    assert server._same_gpu("Intel(R) Arc(TM) A770", "Intel Arc A770")
    assert not server._same_gpu("NVIDIA GeForce RTX 3060", "NVIDIA GeForce RTX 3060 Ti")
    assert not server._same_gpu("AMD Radeon RX 7800 XT", "AMD Radeon RX 7800")
    assert not server._same_gpu("GPU 0", "GPU 1")


def test_luid_key_ignores_padding() -> None:
    assert server._luid_key_from_text(r"luid_0x00000000_0x00013CE1_phys_0") == "0_13ce1"


def test_intel_parser_uses_mib_field_names() -> None:
    payload = json.dumps({
        "devices": [{
            "device_name": "Intel Arc A770",
            "GPU Utilization (%)": "12 %",
            "Memory Used (MiB)": 2048,
            "Memory Total (MiB)": 16384,
            "GPU Temperature (C)": 41,
            "Graphics Clock": "2100 MHz",
        }]
    })
    gpu = server._parse_intel_json(payload)[0]
    assert gpu["name"] == "Intel Arc A770"
    assert gpu["gpu_util_percent"] == 12.0
    assert gpu["vram_used_gb"] == 2.0
    assert gpu["vram_total_gb"] == 16.0
    assert gpu["temperature_c"] == 41.0
    assert gpu["graphics_clock_mhz"] == 2100
    assert gpu["source"] == "intel"
    assert gpu["power_watts"] is None


def test_amd_zero_used_bytes_stay_zero() -> None:
    payload = json.dumps([{
        "market_name": "AMD Radeon RX 7800 XT",
        "gfx_activity": 0,
        "vram": {"vram_used_bytes": 0, "total": 16384},
        "temperature": {"edge": 0},
    }])
    gpu = server._parse_amd_smi(payload)[0]
    assert gpu["gpu_util_percent"] == 0.0
    assert gpu["vram_used_gb"] == 0.0
    assert gpu["vram_total_gb"] == 16.0
    assert gpu["temperature_c"] == 0.0
    assert gpu["source"] == "amd"


def test_rocm_byte_fields() -> None:
    payload = json.dumps({
        "card0": {
            "Card series": "Radeon RX 7900 XTX",
            "GPU use (%)": "0 %",
            "VRAM Total Used Memory (B)": 0,
            "VRAM Total Memory (B)": 25769803776,
            "Temperature (Sensor edge) (C)": "30",
        }
    })
    gpu = server._parse_rocm_smi(payload)[0]
    assert gpu["vram_used_gb"] == 0.0
    assert gpu["vram_total_gb"] == 24.0
    assert gpu["temperature_c"] == 30.0
    assert gpu["gpu_util_percent"] == 0.0


def test_windows_join_matches_luid_not_list_index() -> None:
    controllers = [
        {"name": "NVIDIA GeForce RTX 3060", "adapter_ram": 0xFFFFFFFF},
        {"name": "NVIDIA GeForce RTX 3060 Ti", "adapter_ram": 0xFFFFFFFF},
    ]
    luids = [
        {"name": "NVIDIA GeForce RTX 3060 Ti", "luid": "0_2"},
        {"name": "NVIDIA GeForce RTX 3060", "luid": "0_1"},
    ]
    gib = 1024**3
    pdh = {
        "0_1": {"gpu_util_percent": 10, "vram_used_bytes": gib, "vram_total_bytes": 12 * gib},
        "0_2": {"gpu_util_percent": 80, "vram_used_bytes": 2 * gib, "vram_total_bytes": 8 * gib},
    }
    gpus, warnings = server._join_windows_adapters(controllers, luids, [], pdh)
    assert [gpu["gpu_util_percent"] for gpu in gpus] == [10.0, 80.0]
    assert gpus[0]["vram_total_gb"] == 12.0
    assert gpus[1]["vram_total_gb"] == 8.0
    assert not any("aligned by index" in warning for warning in warnings)
    assert not any("AdapterRAM" in warning for warning in warnings)


def test_windows_join_uses_index_only_when_counts_match() -> None:
    controllers = [
        {"name": "GPU Alpha", "adapter_ram": None},
        {"name": "GPU Beta", "adapter_ram": None},
    ]
    pdh = {
        "0_1": {"gpu_util_percent": 1, "vram_used_bytes": None, "vram_total_bytes": None},
        "0_2": {"gpu_util_percent": 2, "vram_used_bytes": None, "vram_total_bytes": None},
    }
    gpus, warnings = server._join_windows_adapters(controllers, [], [], pdh)
    assert [gpu["gpu_util_percent"] for gpu in gpus] == [1.0, 2.0]
    assert any("aligned by index" in warning for warning in warnings)


def test_windows_join_does_not_guess_when_counts_differ() -> None:
    controllers = [
        {"name": "NVIDIA GeForce RTX 3060", "adapter_ram": None},
        {"name": "Intel UHD Graphics 770", "adapter_ram": None},
    ]
    pdh = {
        "0_9": {"gpu_util_percent": 50, "vram_used_bytes": None, "vram_total_bytes": None},
    }
    gpus, warnings = server._join_windows_adapters(controllers, [], [], pdh)
    assert [gpu["gpu_util_percent"] for gpu in gpus] == [None, None]
    assert any("left empty" in warning for warning in warnings)


def test_merge_keeps_distinct_models_and_vendor_priority() -> None:
    collected: list[dict] = []
    server._merge_gpu(collected, server._gpu(name="NVIDIA GeForce RTX 3060", source="nvidia"))
    server._merge_gpu(collected, server._gpu(name="NVIDIA GeForce RTX 3060 Ti", source="nvidia"))
    server._merge_gpu(collected, server._gpu(name="RTX 3060", source="windows"))
    server._merge_gpu(collected, server._gpu(name="Intel UHD Graphics 770", source="windows"))
    assert [gpu["name"] for gpu in collected] == [
        "NVIDIA GeForce RTX 3060",
        "NVIDIA GeForce RTX 3060 Ti",
        "Intel UHD Graphics 770",
    ]
    assert collected[0]["source"] == "nvidia"


def test_powershell_counter_script_is_balanced() -> None:
    script = server._GPU_COUNTER_SCRIPT
    assert script.count("{") == script.count("}")
    assert "Dedicated Limit" in script
    assert "foreach ($sample in @($samples.CounterSamples))" in script


def test_inventory_script_is_balanced_aside_from_the_class_guid() -> None:
    script = server._INVENTORY_SCRIPT.replace("{4d36e968-e325-11ce-bfc1-08002be10318}", "GPU_CLASS")
    assert script.count("{") == script.count("}")
    assert "HardwareInformation.qwMemorySize" in script
    assert "AdapterLuid" in script


def test_disk_record_has_path_and_percent() -> None:
    record = server._disk_record(".")
    assert record is not None
    assert {"path", "free_gb", "total_gb", "percent", "fstype"} <= set(record)
    assert record["total_gb"] >= record["free_gb"]
