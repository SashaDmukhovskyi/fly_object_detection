# YOLO + MAVLink + ArduPilot SITL Guidance

Цей проєкт підключає YOLO-модель до ArduPilot SITL.

## Що робить

1. Підключається до ArduPilot SITL через MAVLink.
2. Переводить дрон у GUIDED.
3. Робить ARM FORCE для симулятора.
4. Робить TAKEOFF на 5 метрів.
5. Відкриває відео.
6. YOLO шукає ціль у відео.
7. Рахує dx/dy відносно центру кадру.
8. Формує швидкості руху.
9. Надсилає команди в ArduPilot SITL.
10. QGroundControl показує рух дрона на карті.
11. Після завершення виконує LAND.

## Важливо

Цей скрипт зроблений для SITL. Він перевіряє, що підключення тільки локальне:

```python
udp:127.0.0.1:14551
```

Не використовуй його на реальному дроні без окремого безпечного тестування.

## Перед запуском

Запусти ArduPilot SITL з окремим портом для Python:

```bash
cd ~/autopilot/ardupilot
python3 Tools/autotest/sim_vehicle.py -v ArduCopter --console --map --out=udp:127.0.0.1:14551
```

QGroundControl можна залишити відкритим.

## Структура

```text
yolo_mavlink_sitl_guidance/
│
├── yolo_mavlink_sitl_guidance.py
├── requirements.txt
├── README.md
├── models/
│   └── best.pt
└── videos/
    └── test_video.avi
```

## Як підготувати

Створи папки:

```bash
mkdir -p models videos
```

Поклади модель:

```text
models/best.pt
```

Поклади відео:

```text
videos/test_video.avi
```

Або зміни шляхи у файлі `yolo_mavlink_sitl_guidance.py`.

## Встановлення

```bash
pip install -r requirements.txt
```

## Запуск

```bash
python3 yolo_mavlink_sitl_guidance.py
```

## Що побачиш

- у QGroundControl: рух точки/дрона на карті;
- у OpenCV-вікні: відео, рамка YOLO, центр цілі, dx/dy, команда руху;
- у терміналі: підключення, takeoff, команди, посадка.
