[English](README.md)

# mcp-system-info

Локальный MCP-сервер на Python (официальная библиотека `mcp` 2.x, `MCPServer`)
с инструментом `get_system_info`: CPU, диск, RAM и GPU (NVIDIA / Intel Arc / AMD).

## Структура

- `server.py` — MCP-сервер (stdio)
- `client.py` — тестовый клиент
- `requirements.txt` — ставит проект; зависимости объявлены в `pyproject.toml`
- `pyproject.toml` — пакет и точка входа `mcp-system-info`
- `tests/` — тесты парсеров (`pytest`)

## GPU: каскад сбора метрик

Для каждой найденной карты сервер старается вернуть:

- `name` — имя адаптера
- `gpu_util_percent` — загрузка ядра GPU (%)
- `vram_used_gb` / `vram_total_gb` — занятая / общая видеопамять (ГБ)
- `temperature_c` — температура (°C) или `null`, если недоступна
- `graphics_clock_mhz` / `memory_clock_mhz` — частоты, если источник их отдаёт
- `power_watts` / `fan_percent` / `power_limit_watts` — питание и вентилятор (NVML; иначе `null`)
- `source` — откуда взяты данные: `nvidia` | `intel` | `amd` | `windows`

Порядок сбора:

1. **NVIDIA** — `nvidia-ml-py` (`pynvml`): утилизация, VRAM, температура.
2. **Intel Arc** — если в PATH есть `xpu-smi` (Intel oneAPI / IGCL), парсится JSON-вывод.
3. **AMD** — `amd-smi` / `rocm-smi` (JSON). На Windows без утилит имя и объём берутся из реестра адаптера (`HardwareInformation.qwMemorySize`) и LUID DirectX; WMI `AdapterRAM` — только запасной вариант, он врёт на картах больше 4 ГБ.
4. **Счётчики Windows** — GPU Engine и Adapter Memory (`Dedicated Usage`, `Dedicated Limit`) через `win32pdh`, либо PowerShell `Get-Counter`, если PDH недоступен. Счётчики привязываются к адаптеру по LUID, а не по порядку в списке. Температура здесь обычно недоступна → `null`.

Вендорные результаты объединяются. Windows затем добавляет только те адаптеры, которых ещё нет: встроенная Intel рядом с дискретной NVIDIA остаётся в ответе. `RTX 3060` и `RTX 3060 Ti` не склеиваются.

Помимо GPU ответ содержит хост, модель CPU, число ядер, uptime, системный диск (`disk_path`, `disk_percent`) и список томов `disks`. Поле `warnings` описывает сбои сбора. Повторный вызов в пределах 1.5 с возвращает тот же снимок.

Аргумент `fast: true` пропускает блокирующий замер CPU и второй сэмпл счётчиков GPU. Загрузка при этом может быть `null`, объёмы памяти остаются точными. Проверка: `python client.py --fast`.

## Быстрый старт через `uvx`

Одна команда без локальной установки (нужен [uv](https://docs.astral.sh/uv/)):

```bash
uvx --from git+https://github.com/ZetsuOne/mcp-system-info mcp-system-info
```

Конфиг для Cursor / Claude Desktop:

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

## Пошаговая установка и проверка (Windows / PowerShell)

### 1. Перейти в папку проекта

```powershell
cd C:\path\to\mcp-system-info
```

### 2. Создать виртуальное окружение

```powershell
python -m venv .venv
```

### 3. Активировать venv

```powershell
.\.venv\Scripts\Activate.ps1
```

Если PowerShell запрещает скрипты, один раз выполните:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 4. Установить зависимости

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Опционально для расширенных метрик:

- NVIDIA: драйвер NVIDIA (NVML идёт с драйвером; пакет `nvidia-ml-py` уже в `requirements.txt`)
- Intel Arc: установите [Intel XPU System Management Interface](https://www.intel.com/) (`xpu-smi` в PATH)
- AMD: `amd-smi` (Adrenalin) или `rocm-smi` (ROCm)

Без утилит Intel/AMD сервер всё равно работает: NVIDIA через NVML, остальные карты — через Windows fallback.

### 5. Запустить проверку

```powershell
python client.py
```

Ожидаемый вывод — JSON с `cpu_percent`, `disk_*`, `ram_*` и массивом `gpus`.

### 6. (Опционально) Запустить только сервер

```powershell
python server.py
```

Сервер ждёт MCP-клиент на stdin/stdout и сам по себе «молчит» — это нормально.

## Подключение к Cursor (опционально)

```json
{
  "mcpServers": {
    "system-info": {
      "command": "C:\\path\\to\\mcp-system-info\\.venv\\Scripts\\python.exe",
      "args": [
        "C:\\path\\to\\mcp-system-info\\server.py"
      ]
    }
  }
}
```
