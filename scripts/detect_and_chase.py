import asyncio
import math
import numpy as np
import cv2
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw


# =========================
# CONSTANTS
# =========================
SPHERE_RADIUS = 0.5  # meters (1m diameter)
IMAGE_W = 1024
IMAGE_H = 768
HFOV = math.radians(107)
FOCAL_LEN = (IMAGE_W / 2) / math.tan(HFOV / 2)
CAMERA_TILT = math.radians(90)  # straight up
HIT_DISTANCE = 1.0  # meters — close enough to count as a hit

# Velocity control gains
KP_CENTER = 3.0        # m/s per normalized pixel error
KP_APPROACH = 2.0      # m/s upward approach speed
MAX_CENTER_VEL = 4.0   # max lateral velocity
LUNGE_DISTANCE = 5.0   # meters — start lunging when this close
LUNGE_SPEED = 5.0      # m/s — velocity toward sphere during lunge

# Takeoff
TAKEOFF_SPEED = 2.0    # m/s upward
TAKEOFF_TIME = 5.0     # seconds to climb

# Search
SEARCH_DESCEND = 0.3   # m/s descend during search
SEARCH_SPEED = 1.5     # m/s lateral during search
SEARCH_OMEGA = 0.8     # rad/s angular speed of circle
LOST_FRAMES_TO_SEARCH = 15


# =========================
# STATE
# =========================
drone_yaw_deg = 0.0
lost_frames = 0
search_time = 0.0


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

    if cv2.contourArea(c) < 50:
        return None

    (cx, cy), radius = cv2.minEnclosingCircle(c)
    cx, cy, radius = int(cx), int(cy), float(radius)

    if radius < 3:
        return None

    return cx, cy, radius


# =========================
# TELEMETRY
# =========================
async def subscribe_telemetry(drone):
    global drone_yaw_deg

    async for att in drone.telemetry.attitude_euler():
        drone_yaw_deg = att.yaw_deg


# =========================
# MAIN
# =========================
async def run():
    global lost_frames, search_time

    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected")
            break

    # Start telemetry
    telemetry_task = asyncio.create_task(subscribe_telemetry(drone))

    # Wait for EKF to initialize (heading from magnetometer)
    print("Waiting for EKF init...")
    await asyncio.sleep(10)

    # Prime offboard with zero velocity
    print("Priming offboard...")
    for _ in range(20):
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
        await asyncio.sleep(0.1)

    print("Starting offboard...")
    await drone.offboard.start()

    # Arm with retries
    print("Arming...")
    for attempt in range(10):
        try:
            await drone.action.arm()
            print("Armed")
            break
        except Exception as e:
            print(f"Arm attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2)

    # Takeoff: climb at fixed speed
    print("Taking off...")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, -TAKEOFF_SPEED, 0))
    await asyncio.sleep(TAKEOFF_TIME)

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

            result = detect_red_object(frame)

            if result:
                cx, cy, radius_px = result
                lost_frames = 0
                search_time = 0.0

                # Pixel error normalized to [-1, 1]
                ex = (cx - IMAGE_W / 2) / (IMAGE_W / 2)
                ey = (IMAGE_H / 2 - cy) / (IMAGE_H / 2)

                # Distance estimate from apparent size
                dist_est = (SPHERE_RADIUS * FOCAL_LEN) / radius_px

                # Centering velocities (camera up, yaw=0: ey→north, ex→east)
                vn = KP_CENTER * ey
                ve = KP_CENTER * ex

                if dist_est < LUNGE_DISTANCE:
                    vd = -LUNGE_SPEED
                else:
                    vd = -KP_APPROACH

                # Clamp lateral velocity
                mag = math.sqrt(vn**2 + ve**2)
                if mag > MAX_CENTER_VEL:
                    scale = MAX_CENTER_VEL / mag
                    vn *= scale
                    ve *= scale

                await drone.offboard.set_velocity_ned(
                    VelocityNedYaw(vn, ve, vd, 0.0)
                )

                # Visualization
                cv2.circle(frame, (cx, cy), int(radius_px), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                cv2.circle(frame, (IMAGE_W // 2, IMAGE_H // 2), 4, (0, 0, 255), -1)

                cv2.putText(frame, f"dist: {dist_est:.1f}m  vel: ({vn:.1f},{ve:.1f},{vd:.1f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if dist_est < HIT_DISTANCE:
                    cv2.putText(frame, "HIT!", (IMAGE_W // 2 - 50, IMAGE_H // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
                    cv2.imshow("Tracking", frame)
                    cv2.waitKey(3000)
                    break
            else:
                lost_frames += 1

                if lost_frames < LOST_FRAMES_TO_SEARCH:
                    # Brief loss: hover
                    await drone.offboard.set_velocity_ned(
                        VelocityNedYaw(0, 0, 0, 0)
                    )
                else:
                    # Search: fly in circles + descend
                    search_time += 1.0 / 30.0
                    vn = SEARCH_SPEED * math.cos(SEARCH_OMEGA * search_time)
                    ve = SEARCH_SPEED * math.sin(SEARCH_OMEGA * search_time)
                    vd = SEARCH_DESCEND

                    await drone.offboard.set_velocity_ned(
                        VelocityNedYaw(vn, ve, vd, 0.0)
                    )
                    cv2.putText(frame, "SEARCHING...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.imshow("Tracking", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        telemetry_task.cancel()
        print("Stopping...")
        await drone.offboard.stop()
        await drone.action.disarm()


asyncio.run(run())
