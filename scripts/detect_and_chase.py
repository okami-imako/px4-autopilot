import asyncio
import math
import numpy as np
import cv2
from mavsdk import System
from mavsdk.offboard import PositionNedYaw


# =========================
# CONSTANTS
# =========================
SPHERE_RADIUS = 0.5  # meters (1m diameter)
IMAGE_W = 1024
IMAGE_H = 768
HFOV = math.radians(107)
FOCAL_LEN = (IMAGE_W / 2) / math.tan(HFOV / 2)  # ~485.9 px
HIT_DISTANCE = 1.0  # meters — close enough to count as a hit
CLIMB_RATE = 0.5    # m/s — gradual ascent toward sphere
SEARCH_YAW_RATE = 30.0   # deg/s — spin rate when searching
SEARCH_DESCEND = 0.3     # m/s — descend while searching (widens camera coverage)
LOST_FRAMES_TO_SEARCH = 15  # frames without detection before entering search mode


# =========================
# STATE
# =========================
drone_north = 0.0
drone_east = 0.0
drone_down = 0.0
drone_yaw_deg = 0.0

target_north = 0.0
target_east = 0.0
target_down = 0.0
lost_frames = 0
search_yaw = 0.0


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
# POSITION ESTIMATION
# =========================
def compute_sphere_position(cx, cy, radius_px, d_north, d_east, d_down, yaw_deg):
    """Estimate sphere world position in NED from pixel detection + drone telemetry."""
    yaw_rad = math.radians(yaw_deg)

    # Distance along optical axis (vertical for upward camera)
    dist_vertical = (SPHERE_RADIUS * FOCAL_LEN) / radius_px

    # Pixel offset from image center
    dx_px = cx - IMAGE_W / 2
    dy_px = cy - IMAGE_H / 2

    # Lateral offsets in body frame (camera pointing up, pitch=-90°)
    # Image +x -> body +Y (East at yaw=0)
    # Image +y -> body +X (North at yaw=0)
    offset_east_body = dist_vertical * dx_px / FOCAL_LEN
    offset_north_body = dist_vertical * dy_px / FOCAL_LEN

    # Rotate by drone yaw to NED
    offset_north = offset_north_body * math.cos(yaw_rad) - offset_east_body * math.sin(yaw_rad)
    offset_east = offset_north_body * math.sin(yaw_rad) + offset_east_body * math.cos(yaw_rad)

    # Sphere position in NED (sphere is above drone)
    sphere_north = d_north + offset_north
    sphere_east = d_east + offset_east
    sphere_down = d_down - dist_vertical

    return sphere_north, sphere_east, sphere_down


# =========================
# TELEMETRY
# =========================
async def subscribe_telemetry(drone):
    global drone_north, drone_east, drone_down, drone_yaw_deg

    async def update_position():
        global drone_north, drone_east, drone_down
        async for pvn in drone.telemetry.position_velocity_ned():
            drone_north = pvn.position.north_m
            drone_east = pvn.position.east_m
            drone_down = pvn.position.down_m

    async def update_attitude():
        global drone_yaw_deg
        async for att in drone.telemetry.attitude_euler():
            drone_yaw_deg = att.yaw_deg

    await asyncio.gather(update_position(), update_attitude())


# =========================
# MAIN
# =========================
async def run():
    global target_north, target_east, target_down, lost_frames, search_yaw

    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected")
            break

    async for health in drone.telemetry.health():
        if health.is_armable:
            print("Armable")
            break

    # Start telemetry before arming
    telemetry_task = asyncio.create_task(subscribe_telemetry(drone))
    await asyncio.sleep(1)

    print("Arming...")
    await drone.action.arm()

    print("Takeoff...")
    await drone.action.takeoff()
    await asyncio.sleep(5)

    # Initialize targets to current drone position
    target_north = drone_north
    target_east = drone_east
    target_down = drone_down

    # Prime offboard with current position
    initial_pos = PositionNedYaw(drone_north, drone_east, drone_down, 0.0)
    for _ in range(10):
        await drone.offboard.set_position_ned(initial_pos)
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

            result = detect_red_object(frame)

            if result:
                cx, cy, radius_px = result
                lost_frames = 0

                sn, se, sd = compute_sphere_position(
                    cx, cy, radius_px,
                    drone_north, drone_east, drone_down, drone_yaw_deg
                )

                # Track laterally, but climb gradually (don't jump to sphere altitude)
                target_north = sn
                target_east = se
                dist_vertical = (SPHERE_RADIUS * FOCAL_LEN) / radius_px
                if dist_vertical > HIT_DISTANCE:
                    target_down = drone_down - CLIMB_RATE / 30.0
                else:
                    target_down = sd

                await drone.offboard.set_position_ned(
                    PositionNedYaw(target_north, target_east, target_down, 0.0)
                )

                # visualization
                cv2.circle(frame, (cx, cy), int(radius_px), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                cv2.circle(frame, (IMAGE_W // 2, IMAGE_H // 2), 4, (0, 0, 255), -1)

                dist_est = (SPHERE_RADIUS * FOCAL_LEN) / radius_px
                cv2.putText(frame, f"dist: {dist_est:.1f}m  pos: ({sn:.1f},{se:.1f},{sd:.1f})",
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
                    # Brief loss: hold position, don't climb
                    hold_down = max(drone_down, target_down)
                    await drone.offboard.set_position_ned(
                        PositionNedYaw(target_north, target_east, hold_down, 0.0)
                    )
                else:
                    # Search mode: descend slowly + spin to scan sky
                    search_yaw += SEARCH_YAW_RATE / 30.0
                    search_down = drone_down + SEARCH_DESCEND / 30.0
                    await drone.offboard.set_position_ned(
                        PositionNedYaw(target_north, target_east, search_down, search_yaw)
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
