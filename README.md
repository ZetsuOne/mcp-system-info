# mcp-system-info

Локальный MCP-сервер на Python (официальная библиотека `mcp` 2.x, `MCPServer`)
с инструментом `get_system_info`: CPU, диск, RAM и GPU (NVIDIA / Intel Arc / AMD).

## Структура

- `server.py` — MCP-сервер (stdio)
- `client.py` — тестовый клиент
- `requirements.txt` — зависимости

## GPU: каскад сбора метрик

Для каждой найденной карты сервер старается вернуть:

- `name` — имя адаптера
- `gpu_util_percent` — загрузка ядра GPU (%)
- `vram_used_gb` / `vram_total_gb` — занятая / общая видеопамять (ГБ)
- `temperature_c` — температура (°C) или `null`, если недоступна
- `source` — откуда взяты данные: `nvidia` | `intel` | `amd` | `windows`

Порядок сбора:

1. **NVIDIA** — `nvidia-ml-py` (`pynvml`): утилизация, VRAM, температура.
2. **Intel Arc** — если в PATH есть `xpu-smi` (Intel oneAPI / IGCL), парсится JSON-вывод.
3. **AMD** — `amd-smi` / `rocm-smi` (JSON); на Windows без утилит — имя и объём через WMI/`Win32_VideoController`.
4. **Windows fallback** — если вендорные источники ничего не дали: загрузка GPU и dedicated VRAM через счётчики производительности (`win32pdh` / `Get-Counter`). Температура через стандартные API обычно недоступна → безопасно `null`, без падения сервера.

На машинах со смешанными GPU результаты вендорных шагов объединяются (без дублей по имени). Windows-fallback включается только если предыдущие шаги ничего не нашли.

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
