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

# Approach (closer = bigger radius_px = faster)
KP_APPROACH = 0.05        # meters offset per px of radius
MAX_APPROACH_OFFSET = 8.0
MIN_APPROACH_OFFSET = 0.5

# Lateral centering
KP_LATERAL = 4.0          # meters offset per normalized pixel error
MAX_LATERAL_OFFSET = 5.0

# Edge risk
EDGE_ZONE = 0.20          # outer 20% of frame = edge zone
EDGE_FACTOR_MIN = 0.3     # approach multiplier at edge

# Smoothing
EMA_ALPHA = 0.4

# Search
SEARCH_DESCEND = 0.3
SEARCH_RADIUS = 3.0
SEARCH_SPEED = 1.0

# Loss recovery
LOST_FRAMES_BRIEF = 15
LOST_FRAMES_SEARCH = 45
RECOVERY_OFFSET = 2.0


# =========================
# STATE
# =========================
drone_north = 0.0
drone_east = 0.0
drone_down = 0.0
drone_yaw_deg = 0.0

lost_frames = 0
search_time = 0.0
search_center_north = 0.0
search_center_east = 0.0
last_known_north = 0.0
last_known_east = 0.0
last_known_down = 0.0

# Smoothed detection
smooth_cx = IMAGE_W / 2.0
smooth_cy = IMAGE_H / 2.0
smooth_radius = 0.0

# Drift tracking for loss recovery
prev_cx = IMAGE_W / 2.0
prev_cy = IMAGE_H / 2.0
drift_dx = 0.0
drift_dy = 0.0


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
# POSITION ESTIMATION (display only)
# =========================
def compute_sphere_position(cx, cy, radius_px, d_north, d_east, d_down, yaw_deg):
    """Estimate sphere world position in NED using ray casting from tilted camera."""
    yaw_rad = math.radians(yaw_deg)
    cos_t = math.cos(CAMERA_TILT)
    sin_t = math.sin(CAMERA_TILT)

    ray_cam_x = FOCAL_LEN
    ray_cam_y = cx - IMAGE_W / 2
    ray_cam_z = -(cy - IMAGE_H / 2)

    ray_body_x = cos_t * ray_cam_x - sin_t * ray_cam_z
    ray_body_y = ray_cam_y
    ray_body_z = sin_t * ray_cam_x + cos_t * ray_cam_z

    ray_len = math.sqrt(ray_body_x**2 + ray_body_y**2 + ray_body_z**2)
    ray_body_x /= ray_len
    ray_body_y /= ray_len
    ray_body_z /= ray_len

    distance = (SPHERE_RADIUS * FOCAL_LEN) / radius_px

    off_x = distance * ray_body_x
    off_y = distance * ray_body_y
    off_z = distance * ray_body_z

    offset_north = off_x * math.cos(yaw_rad) - off_y * math.sin(yaw_rad)
    offset_east = off_x * math.sin(yaw_rad) + off_y * math.cos(yaw_rad)

    sphere_north = d_north + offset_north
    sphere_east = d_east + offset_east
    sphere_down = d_down - off_z

    return sphere_north, sphere_east, sphere_down


# =========================
# VISUAL SERVOING
# =========================
def smooth_detection(cx, cy, radius_px):
    global smooth_cx, smooth_cy, smooth_radius
    if smooth_radius == 0:
        smooth_cx, smooth_cy, smooth_radius = cx, cy, radius_px
    else:
        smooth_cx = EMA_ALPHA * cx + (1 - EMA_ALPHA) * smooth_cx
        smooth_cy = EMA_ALPHA * cy + (1 - EMA_ALPHA) * smooth_cy
        smooth_radius = EMA_ALPHA * radius_px + (1 - EMA_ALPHA) * smooth_radius
    return smooth_cx, smooth_cy, smooth_radius


def update_drift(cx, cy):
    global prev_cx, prev_cy, drift_dx, drift_dy
    dx = cx - prev_cx
    dy = cy - prev_cy
    drift_dx = 0.3 * dx + 0.7 * drift_dx
    drift_dy = 0.3 * dy + 0.7 * drift_dy
    prev_cx = cx
    prev_cy = cy


def compute_edge_factor(cx, cy):
    nx = abs(cx - IMAGE_W / 2.0) / (IMAGE_W / 2.0)
    ny = abs(cy - IMAGE_H / 2.0) / (IMAGE_H / 2.0)
    max_norm = max(nx, ny)
    if max_norm > (1.0 - EDGE_ZONE):
        t = (max_norm - (1.0 - EDGE_ZONE)) / EDGE_ZONE
        return EDGE_FACTOR_MIN + (1.0 - EDGE_FACTOR_MIN) * (1.0 - t)
    return 1.0


def compute_intercept_target(cx, cy, radius_px, d_n, d_e, d_d, yaw_deg):
    # Normalized pixel errors
    ex = (cx - IMAGE_W / 2.0) / (IMAGE_W / 2.0)
    ey = (IMAGE_H / 2.0 - cy) / (IMAGE_H / 2.0)

    # Lateral offset (center the ball) — camera up: cy→body X, cx→body Y
    off_body_x = KP_LATERAL * ey
    off_body_y = KP_LATERAL * ex

    lat_mag = math.sqrt(off_body_x**2 + off_body_y**2)
    if lat_mag > MAX_LATERAL_OFFSET:
        scale = MAX_LATERAL_OFFSET / lat_mag
        off_body_x *= scale
        off_body_y *= scale

    # Approach offset: proportional to radius_px (closer = faster), gated by edge_factor
    edge_factor = compute_edge_factor(cx, cy)
    approach = KP_APPROACH * radius_px * edge_factor
    approach = max(MIN_APPROACH_OFFSET, min(MAX_APPROACH_OFFSET, approach))
    off_body_z = approach

    # Body to NED
    yaw_rad = math.radians(yaw_deg)
    off_n = off_body_x * math.cos(yaw_rad) - off_body_y * math.sin(yaw_rad)
    off_e = off_body_x * math.sin(yaw_rad) + off_body_y * math.cos(yaw_rad)
    off_d = -off_body_z

    return d_n + off_n, d_e + off_e, d_d + off_d


def reset_servoing_state():
    global smooth_cx, smooth_cy, smooth_radius
    global drift_dx, drift_dy
    smooth_radius = 0.0
    drift_dx = 0.0
    drift_dy = 0.0


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
    global lost_frames, search_time, search_center_north, search_center_east
    global last_known_north, last_known_east, last_known_down

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

    # Initialize last known position
    last_known_north = drone_north
    last_known_east = drone_east
    last_known_down = drone_down

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

                # Smooth detection
                s_cx, s_cy, s_rad = smooth_detection(cx, cy, radius_px)
                update_drift(s_cx, s_cy)

                # Save last known position for search mode
                last_known_north = drone_north
                last_known_east = drone_east
                last_known_down = drone_down

                # Visual servoing: compute intercept target
                tn, te, td = compute_intercept_target(
                    s_cx, s_cy, s_rad,
                    drone_north, drone_east, drone_down, drone_yaw_deg
                )

                await drone.offboard.set_position_ned(
                    PositionNedYaw(tn, te, td, 0.0)
                )

                # Visualization
                cv2.circle(frame, (cx, cy), int(radius_px), (0, 255, 0), 2)
                cv2.circle(frame, (int(s_cx), int(s_cy)), 4, (255, 255, 0), -1)
                cv2.circle(frame, (IMAGE_W // 2, IMAGE_H // 2), 4, (0, 0, 255), -1)
                cv2.line(frame, (IMAGE_W // 2, IMAGE_H // 2),
                         (int(s_cx), int(s_cy)), (0, 255, 255), 1)

                dist_est = (SPHERE_RADIUS * FOCAL_LEN) / s_rad
                edge_f = compute_edge_factor(s_cx, s_cy)
                cv2.putText(frame,
                    f"dist:{dist_est:.1f}m edge:{edge_f:.2f} r_px:{s_rad:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Position estimate for display
                sn, se, sd = compute_sphere_position(
                    s_cx, s_cy, s_rad,
                    drone_north, drone_east, drone_down, drone_yaw_deg
                )
                cv2.putText(frame, f"pos:({sn:.1f},{se:.1f},{sd:.1f})",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Hit detection
                if dist_est < HIT_DISTANCE:
                    cv2.putText(frame, "HIT!", (IMAGE_W // 2 - 50, IMAGE_H // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
                    cv2.imshow("Tracking", frame)
                    cv2.waitKey(3000)
                    break
            else:
                lost_frames += 1

                if lost_frames < LOST_FRAMES_BRIEF:
                    # Brief loss: nudge in drift direction to reacquire
                    if abs(drift_dx) + abs(drift_dy) > 0.5:
                        mag = math.sqrt(drift_dx**2 + drift_dy**2)
                        # drift in pixel space → body offset (same mapping as pixel error)
                        nudge_body_x = RECOVERY_OFFSET * (-drift_dy / mag)
                        nudge_body_y = RECOVERY_OFFSET * (drift_dx / mag)

                        yaw_rad = math.radians(drone_yaw_deg)
                        nudge_n = nudge_body_x * math.cos(yaw_rad) - nudge_body_y * math.sin(yaw_rad)
                        nudge_e = nudge_body_x * math.sin(yaw_rad) + nudge_body_y * math.cos(yaw_rad)

                        await drone.offboard.set_position_ned(
                            PositionNedYaw(drone_north + nudge_n,
                                           drone_east + nudge_e,
                                           drone_down, 0.0)
                        )
                    else:
                        # No clear drift — hover
                        await drone.offboard.set_position_ned(
                            PositionNedYaw(drone_north, drone_east, drone_down, 0.0)
                        )

                    cv2.putText(frame, "LOST (recovering...)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                elif lost_frames < LOST_FRAMES_SEARCH:
                    # Hover before committing to search
                    await drone.offboard.set_position_ned(
                        PositionNedYaw(drone_north, drone_east, drone_down, 0.0)
                    )
                    cv2.putText(frame, "LOST (hovering...)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                else:
                    # Extended loss: spiral search around last known position
                    if lost_frames == LOST_FRAMES_SEARCH:
                        search_center_north = last_known_north
                        search_center_east = last_known_east
                        reset_servoing_state()

                    search_time += 1.0 / 30.0
                    spiral_n = search_center_north + SEARCH_RADIUS * math.cos(SEARCH_SPEED * search_time)
                    spiral_e = search_center_east + SEARCH_RADIUS * math.sin(SEARCH_SPEED * search_time)
                    search_down = last_known_down + SEARCH_DESCEND / 30.0
                    last_known_down = search_down

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
