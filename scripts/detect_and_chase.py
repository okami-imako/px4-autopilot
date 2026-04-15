import asyncio
import numpy as np
import cv2
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw


# =========================
# STATE (very important)
# =========================
filtered_x = 0.0
filtered_y = 0.0


# =========================
# VISION
# =========================
def detect_red_object(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 120, 80])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 120, 80])
    upper2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)

    if cv2.contourArea(c) < 50:  # ignore noise (important for 1–2% object)
        return None

    x, y, w, h = cv2.boundingRect(c)
    cx = x + w // 2
    cy = y + h // 2

    return cx, cy, x, y, w, h


# =========================
# CONTROL (stable baseline)
# =========================
def compute_velocity(cx, cy, width, height):
    global filtered_x, filtered_y

    # normalize error (-1..1)
    ex = (cx - width / 2) / (width / 2)
    ey = (cy - height / 2) / (height / 2)

    # VERY IMPORTANT: heavy smoothing
    ALPHA = 0.08
    filtered_x = (1 - ALPHA) * filtered_x + ALPHA * ex
    filtered_y = (1 - ALPHA) * filtered_y + ALPHA * ey

    DEADZONE = 0.01

    def apply_deadzone(x):
        if abs(x) < DEADZONE:
            return 0.0
        # smooth transition instead of hard cutoff
        return np.sign(x) * (abs(x) - DEADZONE) / (1 - DEADZONE)


    filtered_x = apply_deadzone(filtered_x)
    filtered_y = apply_deadzone(filtered_y)

    # proportional only (NO derivative, NO prediction)
    Kp = 1.0

    vx = Kp * filtered_y
    vy = Kp * filtered_x

    # clamp aggressively (critical for stability)
    vx = np.clip(vx, -1.2, 1.2)
    vy = np.clip(vy, -1.2, 1.2)

    return vx, vy, 0.0, filtered_x, filtered_y


# =========================
# MAIN
# =========================
async def run():
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected")
            break

    # wait armable
    async for health in drone.telemetry.health():
        if health.is_armable:
            print("Armable")
            break

    print("Arming...")
    await drone.action.arm()

    print("Takeoff...")
    await drone.action.takeoff()
    await asyncio.sleep(5)

    # OFFBOARD priming
    for _ in range(10):
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
        )
        await asyncio.sleep(0.05)

    await drone.offboard.start()

    cap = cv2.VideoCapture(
        "udpsrc port=5600 ! application/x-rtp,encoding-name=H264,payload=96 ! "
        "rtph264depay ! decodebin ! videoconvert ! appsink sync=false",
        cv2.CAP_GSTREAMER
    )

    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            h, w, _ = frame.shape

            result = detect_red_object(frame)

            if result:
                cx, cy, x, y, bw, bh = result

                vx, vy, vz, ex, ey = compute_velocity(cx, cy, w, h)

                await drone.offboard.set_velocity_ned(
                    VelocityNedYaw(vx, vy, vz, 0.0)
                )

                # visualization
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                cv2.circle(frame, (w // 2, h // 2), 4, (0, 0, 255), -1)

                cv2.putText(frame, f"err: {ex:.2f},{ey:.2f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255), 2)

            else:
                # hover
                await drone.offboard.set_velocity_ned(
                    VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
                )

            cv2.imshow("Tracking", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        print("Stopping...")
        await drone.offboard.stop()
        await drone.action.disarm()


asyncio.run(run())
