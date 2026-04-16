import asyncio
import math
import numpy as np
import cv2
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, VelocityNedYaw


# =========================
# CONSTANTS
# =========================
SPHERE_RADIUS = 0.5  # meters (1m diameter)
IMAGE_W = 1024
IMAGE_H = 768
HFOV = math.radians(107)
FOCAL_LEN = (IMAGE_W / 2) / math.tan(HFOV / 2)  # ~485.9 px
CAMERA_TILT = math.radians(90)  # straight up
HIT_DISTANCE = 1.0  # meters — close enough to count as a hit
LUNGE_DISTANCE = 5.0  # meters — start lunging when this close
LUNGE_SPEED = 5.0     # m/s — velocity toward sphere during lunge
SEARCH_DESCEND = 0.3     # m/s — descend while searching
SEARCH_RADIUS = 3.0      # meters — spiral search radius
SEARCH_SPEED = 1.0       # rad/s — angular speed of search spiral
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
search_time = 0.0
search_center_north = 0.0
search_center_east = 0.0


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
    """Estimate sphere world position in NED using ray casting from tilted camera."""
    yaw_rad = math.radians(yaw_deg)
    cos_t = math.cos(CAMERA_TILT)
    sin_t = math.sin(CAMERA_TILT)

    # Ray in camera frame (Gazebo: +X optical axis, +Y image right, +Z image up)
    ray_cam_x = FOCAL_LEN
    ray_cam_y = cx - IMAGE_W / 2
    ray_cam_z = -(cy - IMAGE_H / 2)

    # Rotate camera->body by pitch=-CAMERA_TILT (rotation about Y)
    # Body frame: +X forward, +Y right, +Z up
    ray_body_x = cos_t * ray_cam_x - sin_t * ray_cam_z
    ray_body_y = ray_cam_y
    ray_body_z = sin_t * ray_cam_x + cos_t * ray_cam_z

    # Normalize
    ray_len = math.sqrt(ray_body_x**2 + ray_body_y**2 + ray_body_z**2)
    ray_body_x /= ray_len
    ray_body_y /= ray_len
    ray_body_z /= ray_len

    # Distance from camera to sphere center (pinhole model)
    distance = (SPHERE_RADIUS * FOCAL_LEN) / radius_px

    # Sphere offset in body frame
    off_x = distance * ray_body_x  # forward
    off_y = distance * ray_body_y  # right
    off_z = distance * ray_body_z  # up

    # Body to NED (rotate by yaw, flip Z)
    offset_north = off_x * math.cos(yaw_rad) - off_y * math.sin(yaw_rad)
    offset_east = off_x * math.sin(yaw_rad) + off_y * math.cos(yaw_rad)

    sphere_north = d_north + offset_north
    sphere_east = d_east + offset_east
    sphere_down = d_down - off_z  # body +Z up = NED -down

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
    global target_north, target_east, target_down
    global lost_frames, search_time, search_center_north, search_center_east

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
                search_time = 0.0

                sn, se, sd = compute_sphere_position(
                    cx, cy, radius_px,
                    drone_north, drone_east, drone_down, drone_yaw_deg
                )

                dist_est = (SPHERE_RADIUS * FOCAL_LEN) / radius_px

                target_north = sn
                target_east = se
                target_down = sd

                if dist_est < LUNGE_DISTANCE:
                    # Lunge: fly directly at sphere at fixed speed
                    dn = sn - drone_north
                    de = se - drone_east
                    dd = sd - drone_down
                    dist = math.sqrt(dn**2 + de**2 + dd**2)
                    if dist > 0.1:
                        scale = LUNGE_SPEED / dist
                        await drone.offboard.set_velocity_ned(
                            VelocityNedYaw(dn * scale, de * scale, dd * scale, 0.0)
                        )
                    else:
                        await drone.offboard.set_position_ned(
                            PositionNedYaw(target_north, target_east, target_down, 0.0)
                        )
                else:
                    await drone.offboard.set_position_ned(
                        PositionNedYaw(target_north, target_east, target_down, 0.0)
                    )

                # visualization
                cv2.circle(frame, (cx, cy), int(radius_px), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                cv2.circle(frame, (IMAGE_W // 2, IMAGE_H // 2), 4, (0, 0, 255), -1)

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
                    # Brief loss: hold position
                    await drone.offboard.set_position_ned(
                        PositionNedYaw(target_north, target_east, target_down, 0.0)
                    )
                else:
                    # Search mode: spiral around last known position + descend
                    if lost_frames == LOST_FRAMES_TO_SEARCH:
                        search_center_north = target_north
                        search_center_east = target_east

                    search_time += 1.0 / 30.0
                    spiral_n = search_center_north + SEARCH_RADIUS * math.cos(SEARCH_SPEED * search_time)
                    spiral_e = search_center_east + SEARCH_RADIUS * math.sin(SEARCH_SPEED * search_time)
                    search_down = target_down + SEARCH_DESCEND / 30.0
                    target_down = search_down

                    await drone.offboard.set_position_ned(
                        PositionNedYaw(spiral_n, spiral_e, search_down, 0.0)
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
