from pathlib import Path
import time
import cv2
import numpy as np
from ultralytics import YOLO
from pymavlink import mavutil


# =========================
# CONFIG
# =========================

MODEL_PATH = "models/best.pt"
VIDEO_PATH = "videos\test_video.avi" 
 
CONNECTION_STRING = "udp:127.0.0.1:14551"

TAKEOFF_ALTITUDE = 5

CONF = 0.35
DEAD_ZONE = 35

# Швидкість польоту вперед у симуляції.
# 300 км/год = приблизно 83.33 м/с.
# Увага: це значення призначене тільки для ArduPilot SITL.
CRUISE_SPEED_KMH = 300.0
CRUISE_SPEED_MS = CRUISE_SPEED_KMH / 3.6

# Коли ціль не знайдена, дрон не зависає, а летить прямо.
SEARCH_VX = CRUISE_SPEED_MS

# Коли ціль знайдена, дрон також летить вперед, але додає бокове наведення.
TRACK_VX = CRUISE_SPEED_MS

# Максимальна бокова швидкість для супроводу цілі.
# Її не варто робити такою ж великою, як швидкість вперед,
# бо бокове наведення стане занадто різким.
MAX_VY = 20.0   # right/left, m/s
MAX_VZ = 0.0    # висоту в цій версії не змінюємо

# Коефіцієнт перетворення помилки кадру в бокову швидкість.
KP = 0.08

# Коефіцієнт повороту за зміщенням цілі від центра кадру.
# Якщо ціль правіше центра, дрон повертає вправо; якщо лівіше — вліво.
KYAW = 0.0025
MAX_YAW_RATE = 0.8       # rad/s

# Якщо ціль вже була знайдена, але потім зникла з кадру,
# дрон певний час повертає у той бік, куди ціль востаннє змістилась.
LOST_TARGET_TURN_TIME = 6.0   # seconds
LOST_YAW_RATE = 0.25          # rad/s
LOST_SIDE_SPEED = 12.0        # m/s, додатковий рух у бік останньої позиції цілі

# Частота відправлення MAVLink-команд.
COMMAND_PERIOD = 0.1

USE_GPU = False

# Захист: скрипт дозволений тільки для локального SITL
SITL_ONLY = True


# =========================
# SAFETY CHECK
# =========================

def ensure_sitl_connection():
    if SITL_ONLY and "127.0.0.1" not in CONNECTION_STRING:
        raise RuntimeError(
            "Скрипт дозволено запускати тільки з локальним SITL: 127.0.0.1"
        )


# =========================
# MAVLINK FUNCTIONS
# =========================

def connect_vehicle():
    print("Підключення до ArduPilot SITL...")
    master = mavutil.mavlink_connection(CONNECTION_STRING)

    print("Очікування heartbeat...")
    master.wait_heartbeat()

    print(f"Підключено: system={master.target_system}, component={master.target_component}")
    return master


def set_mode(master, mode_name):
    print(f"Зміна режиму на {mode_name}...")

    mode_mapping = master.mode_mapping()
    mode_id = mode_mapping.get(mode_name)

    if mode_id is None:
        raise RuntimeError(f"Режим {mode_name} не знайдено. Доступні: {list(mode_mapping.keys())}")

    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )

    time.sleep(2)
    print(f"Команду режиму {mode_name} відправлено.")


def arm_force(master):
    print("ARM FORCE для SITL...")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,
        21196,
        0,
        0,
        0,
        0,
        0
    )

    master.motors_armed_wait()
    print("Дрон ARMED.")


def takeoff(master, altitude):
    print(f"TAKEOFF на {altitude} м...")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        altitude
    )

    start = time.time()

    while time.time() - start < 40:
        msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=5)

        if msg is None:
            print("Очікування висоти...")
            continue

        current_alt = msg.relative_alt / 1000.0
        print(f"Висота: {current_alt:.1f} м")

        if current_alt >= altitude * 0.90:
            print("Висоту досягнуто.")
            return

        time.sleep(0.5)

    print("Попередження: takeoff timeout, продовжуємо демонстрацію.")


def send_body_velocity(master, vx, vy, vz, yaw_rate=0.0):
    """
    FRAME_BODY_NED:
    vx > 0 — вперед
    vx < 0 — назад
    vy > 0 — вправо
    vy < 0 — вліво
    vz > 0 — вниз

    yaw_rate > 0 — поворот вправо у SITL; якщо у твоїй симуляції
    напрям буде протилежний, поміняй знак у lost_target_command().
    """

    # Використовуємо vx/vy/vz і yaw_rate.
    # Ігноруємо позицію, прискорення та абсолютний yaw.
    type_mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    )

    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        type_mask,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0,
        yaw_rate
    )


def set_param(master, name, value):
    """Зміна параметра ArduPilot у SITL."""
    print(f"SET PARAM {name} = {value}")
    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        name.encode("utf-8"),
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )
    time.sleep(0.15)


def configure_sitl_speed_limits(master):
    """
    У Copter горизонтальна швидкість може обмежуватися параметрами автопілота.
    Без цього MAVLink-команда може містити vx=83 м/с, але SITL фактично
    буде сильно її обрізати або виглядати так, ніби дрон тільки повертає.
    """
    cruise_cm_s = CRUISE_SPEED_MS * 100.0

    # Основний ліміт швидкості навігації Copter задається в см/с.
    # 300 км/год = 83.33 м/с = 8333 см/с.
    for name, value in [
        ("WPNAV_SPEED", cruise_cm_s),
        ("WPNAV_ACCEL", 5000),
        ("WPNAV_ACCEL_C", 5000),
        ("PSC_VELXY_MAX", CRUISE_SPEED_MS),
    ]:
        try:
            set_param(master, name, value)
        except Exception as exc:
            print(f"Не вдалося встановити {name}: {exc}")


def land(master):
    print("LAND...")
    set_mode(master, "LAND")
    time.sleep(2)


# =========================
# VISION / GUIDANCE
# =========================

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def draw_cross(frame, x, y, color, size=12, thickness=2):
    x = int(x)
    y = int(y)
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)


def guidance_from_dxdy(dx, dy, frame_width, frame_height):
    """
    Логіка супроводу:
    - дрон постійно летить вперед із швидкістю TRACK_VX;
    - YOLO знаходить рамку цілі;
    - центр рамки порівнюється з центром кадру;
    - dx перетворюється на бокову швидкість vy і кутову швидкість yaw_rate;
    - якщо ціль майже по центру, боковий рух і поворот не задаються.
    """

    vx = TRACK_VX

    if abs(dx) < DEAD_ZONE:
        vy = 0.0
        yaw_rate = 0.0
    else:
        vy = clamp(dx * KP, -MAX_VY, MAX_VY)
        yaw_rate = clamp(dx * KYAW, -MAX_YAW_RATE, MAX_YAW_RATE)

    # У цій демонстраційній версії висота не змінюється.
    # dy поки використовується тільки для виводу на екран.
    vz = 0.0

    return vx, vy, vz, yaw_rate


def search_forward_command():
    """
    Команда для початкового режиму пошуку.
    Цілі ще не було в кадрі, тому дрон летить прямо без повороту.
    """
    vx = SEARCH_VX
    vy = 0.0
    vz = 0.0
    yaw_rate = 0.0
    return vx, vy, vz, yaw_rate


def lost_target_command(last_target_side, seconds_after_loss):
    """
    Команда для ситуації, коли ціль уже була знайдена,
    але потім вийшла з кадру.

    Важливо: тут дрон НЕ має просто крутитися на місці.
    Тому команда містить одразу три дії:
    - vx > 0: летіти вперед;
    - vy: зміщуватися в останній бік цілі;
    - yaw_rate: плавно довернути камеру/ніс у той самий бік.
    """
    vx = SEARCH_VX
    vz = 0.0

    if seconds_after_loss <= LOST_TARGET_TURN_TIME:
        vy = last_target_side * LOST_SIDE_SPEED
        yaw_rate = last_target_side * LOST_YAW_RATE
    else:
        vy = 0.0
        yaw_rate = 0.0

    return vx, vy, vz, yaw_rate


def get_command_text(vx, vy, vz, yaw_rate=0.0):
    parts = []

    if vx > 0.1:
        parts.append("FORWARD")
    elif vx < -0.1:
        parts.append("BACK")

    if vy > 0.1:
        parts.append("RIGHT")
    elif vy < -0.1:
        parts.append("LEFT")

    if yaw_rate > 0.05:
        parts.append("YAW RIGHT")
    elif yaw_rate < -0.05:
        parts.append("YAW LEFT")

    if not parts:
        return "HOLD"

    return " + ".join(parts)


# =========================
# MAIN
# =========================

def main():
    ensure_sitl_connection()

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Модель не знайдена: {MODEL_PATH}")

    if not Path(VIDEO_PATH).exists():
        raise FileNotFoundError(f"Відео не знайдено: {VIDEO_PATH}")

    print("Завантаження YOLO...")
    model = YOLO(MODEL_PATH)
    print("YOLO завантажено.")

    master = connect_vehicle()
    configure_sitl_speed_limits(master)

    set_mode(master, "GUIDED")
    arm_force(master)
    takeoff(master, TAKEOFF_ALTITUDE)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        land(master)
        raise RuntimeError("Не вдалося відкрити відео.")

    print("YOLO + MAVLink guidance запущено.")
    print("Натисни Q у вікні відео для завершення.")

    last_command_time = 0
    target_was_seen = False
    last_target_side = 1
    last_seen_time = None

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Відео завершилось.")
                break

            height, width = frame.shape[:2]
            cx = width // 2
            cy = height // 2

            predict_kwargs = {
                "source": frame,
                "conf": CONF,
                "verbose": False
            }

            if USE_GPU:
                predict_kwargs["device"] = 0

            try:
                results = model.predict(**predict_kwargs)
            except Exception:
                predict_kwargs.pop("device", None)
                results = model.predict(**predict_kwargs)

            best_box = None
            best_conf = 0.0

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box

            draw_cross(frame, cx, cy, (0, 0, 255), size=15)
            cv2.circle(frame, (cx, cy), DEAD_ZONE, (0, 0, 255), 2)

            if best_box is not None:
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()

                tx = int((x1 + x2) / 2)
                ty = int((y1 + y2) / 2)

                dx = tx - cx
                dy = ty - cy

                # Запам'ятовуємо, що ціль уже була знайдена.
                # Якщо вона зникне з кадру, дрон буде повертати в цей бік.
                target_was_seen = True
                last_seen_time = time.time()
                if abs(dx) >= DEAD_ZONE:
                    last_target_side = 1 if dx > 0 else -1

                vx, vy, vz, yaw_rate = guidance_from_dxdy(dx, dy, width, height)
                command_text = get_command_text(vx, vy, vz, yaw_rate)

                # Надсилаємо MAVLink-команди не частіше 10 разів/сек
                now = time.time()
                if now - last_command_time > COMMAND_PERIOD:
                    send_body_velocity(master, vx, vy, vz, yaw_rate)
                    last_command_time = now

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                draw_cross(frame, tx, ty, (0, 255, 0), size=12)
                cv2.line(frame, (cx, cy), (tx, ty), (255, 255, 0), 2)

                lines = [
                    f"TARGET FOUND conf={best_conf:.2f}",
                    f"dx={dx} dy={dy}",
                    f"MAVLINK CMD: {command_text}",
                    f"speed={CRUISE_SPEED_KMH:.0f} km/h",
                    f"vx={vx:.2f} vy={vy:.2f} vz={vz:.2f}",
                    f"yaw_rate={yaw_rate:.2f} rad/s",
                    "Controller: ArduPilot SITL"
                ]

            else:
                now = time.time()

                if target_was_seen and last_seen_time is not None:
                    seconds_after_loss = now - last_seen_time
                    vx, vy, vz, yaw_rate = lost_target_command(last_target_side, seconds_after_loss)
                    command_text = get_command_text(vx, vy, vz, yaw_rate)
                    search_status = f"LOST TARGET: TURN TO LAST SIDE {seconds_after_loss:.1f}s"
                else:
                    vx, vy, vz, yaw_rate = search_forward_command()
                    command_text = get_command_text(vx, vy, vz, yaw_rate)
                    search_status = "SEARCH MODE: FLY STRAIGHT"

                if now - last_command_time > COMMAND_PERIOD:
                    send_body_velocity(master, vx, vy, vz, yaw_rate)
                    last_command_time = now

                lines = [
                    "TARGET NOT FOUND",
                    search_status,
                    f"MAVLINK CMD: {command_text}",
                    f"speed={CRUISE_SPEED_KMH:.0f} km/h",
                    f"vx={vx:.2f} vy={vy:.2f} vz={vz:.2f}",
                    f"yaw_rate={yaw_rate:.2f} rad/s",
                    "Controller: ArduPilot SITL"
                ]

            for i, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (20, 40 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            cv2.imshow("YOLO + ArduPilot SITL Guidance", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        land(master)
        print("Демонстрацію завершено.")


if __name__ == "__main__":
    main()