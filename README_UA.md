# Сервіс розпізнавання об'єктів (CPU)

REST API сервіс для розпізнавання об'єктів на зображеннях з використанням FastAPI та YOLOv5.

## Можливості

- Робота тільки на CPU
- Локальне зберігання моделей
- Заміна моделей через API
- Підтримка власних моделей YOLOv5
- Налаштування розміру зображення для інференсу (320-1280)
- Регульована товщина рамок
- Пакетна обробка зображень

## Швидкий старт

### 1. Підготовка моделі

Завантажте модель YOLOv5 та помістіть у папку `models/`:

```bash
mkdir -p models
# Завантажити стандартну модель
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt -O models/yolov5s.pt
```

Доступні моделі:
- `yolov5n.pt` - nano (найшвидша, найменш точна)
- `yolov5s.pt` - small
- `yolov5m.pt` - medium
- `yolov5l.pt` - large
- `yolov5x.pt` - extra large (найточніша, найповільніша)

### 2. Запуск

**Локально:**
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Docker:**
```bash
docker-compose up --build
```

## API Ендпоінти

### Детекція

| Ендпоінт | Метод | Опис |
|----------|-------|------|
| `/detect` | POST | JSON з результатами детекції |
| `/detect/image` | POST | Зображення з рамками |
| `/detect/batch` | POST | Пакетна обробка |
| `/classes` | GET | Список класів моделі |
| `/sizes` | GET | Доступні розміри інференсу |

### Керування моделями

| Ендпоінт | Метод | Опис |
|----------|-------|------|
| `/models` | GET | Список всіх моделей |
| `/models/current` | GET | Інформація про активну модель |
| `/models/upload` | POST | Завантажити нову модель |
| `/models/activate/{name}` | POST | Активувати модель |
| `/models/{name}` | DELETE | Видалити модель |

## Параметри

### Параметри детекції

| Параметр | Тип | За замовчуванням | Опис |
|----------|-----|------------------|------|
| `confidence_threshold` | float | 0.25 | Поріг впевненості (0-1) |
| `imgsz` | string | "640" | Розмір зображення для інференсу |
| `line_thickness` | int | 2 | Товщина рамки (1-10) |

### Доступні розміри зображення

| Розмір | Опис |
|--------|------|
| 320 | Швидкий, низька якість |
| 416 | Швидкий |
| 512 | Середній |
| 640 | Стандартний (за замовчуванням) |
| 768 | Висока якість |
| 1024 | Дуже висока якість |
| 1280 | Максимальна якість, повільний |

## Приклади використання

### Завантаження нової моделі

```bash
# Завантажити та активувати одразу
curl -X POST "http://localhost:8000/models/upload?activate=true" \
  -F "file=@yolov5m.pt"

# Тільки завантажити (без активації)
curl -X POST "http://localhost:8000/models/upload?activate=false" \
  -F "file=@yolov5l.pt"
```

### Перемикання між моделями

```bash
# Список моделей
curl http://localhost:8000/models

# Активувати іншу модель
curl -X POST "http://localhost:8000/models/activate/yolov5m.pt"

# Перевірити поточну модель
curl http://localhost:8000/models/current
```

### Детекція об'єктів

```bash
# Отримати JSON результат
curl -X POST "http://localhost:8000/detect?imgsz=640&confidence_threshold=0.5" \
  -F "file=@image.jpg"

# Отримати зображення з рамками
curl -X POST "http://localhost:8000/detect/image?imgsz=768&line_thickness=3" \
  -F "file=@image.jpg" \
  --output result.jpg
```

### Python клієнт

```python
import requests

API = "http://localhost:8000"

# Завантаження моделі
with open("my_custom_model.pt", "rb") as f:
    resp = requests.post(
        f"{API}/models/upload",
        files={"file": f},
        params={"activate": "true"}
    )
    print(resp.json())

# Детекція
with open("image.jpg", "rb") as f:
    resp = requests.post(
        f"{API}/detect",
        files={"file": f},
        params={
            "confidence_threshold": 0.5,
            "imgsz": "640"
        }
    )
    print(resp.json())

# Отримати зображення з рамками
with open("image.jpg", "rb") as f:
    resp = requests.post(
        f"{API}/detect/image",
        files={"file": f},
        params={
            "confidence_threshold": 0.5,
            "imgsz": "1024",
            "line_thickness": 3
        }
    )
    with open("result.jpg", "wb") as out:
        out.write(resp.content)
```

## Структура проекту

```
object_detection_service/
├── main.py              # FastAPI додаток
├── requirements.txt     # Залежності
├── Dockerfile
├── docker-compose.yml
├── client.py            # Приклад клієнта
├── models/              # Директорія для моделей
│   ├── yolov5s.pt       # Модель YOLOv5
│   └── active_model.pt  # Копія активної моделі
├── README.md            # Документація (English)
└── README_UA.md         # Документація (Українська)
```

## Власні моделі

Сервіс підтримує власні моделі YOLOv5, навчені на ваших даних.

**Вимоги до моделі:**
- Формат PyTorch (.pt)
- Архітектура YOLOv5
- Сумісність з `torch.hub.load("ultralytics/yolov5", "custom", ...)`

**Завантаження власної моделі:**
```bash
curl -X POST "http://localhost:8000/models/upload" \
  -F "file=@my_custom_model.pt" \
  -F "activate=true"
```

## Продуктивність (CPU)

| Модель | Розмір | Час інференсу* |
|--------|--------|----------------|
| yolov5n | 4 MB | ~50мс |
| yolov5s | 14 MB | ~100мс |
| yolov5m | 42 MB | ~200мс |
| yolov5l | 92 MB | ~350мс |
| yolov5x | 174 MB | ~500мс |

*Intel Core i7, зображення 640x640

## Документація API

Після запуску сервера доступна інтерактивна документація:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Ліцензія

MIT
