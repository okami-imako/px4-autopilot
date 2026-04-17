import asyncio
import math
import time
import numpy as np
import cv2
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw


# =========================
# CONSTANTS
# =========================

# Camera & geometry
SPHERE_RADIUS = 0.5          # meters (1m diameter)
IMAGE_W = 1024
IMAGE_H = 768
HFOV = math.radians(107)
FOCAL_LEN = (IMAGE_W / 2) / math.tan(HFOV / 2)
HIT_DISTANCE = 1.0           # meters — close enough to count as a hit

# Velocity command smoothing
CMD_SMOOTH = 0.5             # EMA alpha (lower = smoother)

# Lateral control — feedforward + bearing-proportional with cap
K_LATERAL = 4.0              # m/s per radian of bearing offset from vertical
MAX_LATERAL_VEL = 12.0        # m/s — room for feedforward + correction

# Vertical speed profile (within MPC_Z_VEL_MAX_UP = 3.0 default)
APPROACH_FAR_DIST = 10.0
APPROACH_CLOSE_DIST = 3.0
APPROACH_FAR_SPEED = 2.5     # m/s upward — far zone
APPROACH_CLOSE_SPEED = 3.0   # m/s upward — close/terminal

# Tracker (EMA velocity estimation)
EMA_ALPHA = 0.2              # smoother NED bearing rate
MAX_INTERCEPT_TIME = 1.0     # seconds — moderate prediction for correction direction
MIN_DT = 0.005
MAX_DT = 0.2

# Takeoff
TAKEOFF_SPEED = 2.0
TAKEOFF_TIME = 5.0

# Search (when target lost)
SEARCH_ASCEND_SPEED = 1.0
SEARCH_LATERAL_SPEED = 1.5
SEARCH_OMEGA = 0.8
LOST_PREDICT_FRAMES = 10
LOST_DECEL_FRAMES = 30


# =========================
# ROTATION HELPERS
# =========================
def body_to_ned(bx, by, bz, roll_deg, pitch_deg, yaw_deg):
    """Rotate body-frame vector to NED using ZYX Euler convention.
    PX4 body: X forward, Y right, Z down.
    """
    phi = math.radians(roll_deg)
    theta = math.radians(pitch_deg)
    psi = math.radians(yaw_deg)
    cp, sp = math.cos(phi), math.sin(phi)
    ct, st = math.cos(theta), math.sin(theta)
    cy, sy = math.cos(psi), math.sin(psi)

    n = cy * ct * bx + (cy * st * sp - sy * cp) * by + (cy * st * cp + sy * sp) * bz
    e = sy * ct * bx + (sy * st * sp + cy * cp) * by + (sy * st * cp - cy * sp) * bz
    d = -st * bx + ct * sp * by + ct * cp * bz
    return n, e, d


def ned_to_body(n, e, d, roll_deg, pitch_deg, yaw_deg):
    """Rotate NED vector to body frame (transpose of body_to_ned)."""
    phi = math.radians(roll_deg)
    theta = math.radians(pitch_deg)
    psi = math.radians(yaw_deg)
    cp, sp = math.cos(phi), math.sin(phi)
    ct, st = math.cos(theta), math.sin(theta)
    cy, sy = math.cos(psi), math.sin(psi)

    bx = cy * ct * n + sy * ct * e - st * d
    by = (cy * st * sp - sy * cp) * n + (sy * st * sp + cy * cp) * e + ct * sp * d
    bz = (cy * st * cp + sy * sp) * n + (sy * st * cp - cy * sp) * e + ct * cp * d
    return bx, by, bz


def pixel_to_ned_bearing(cx, cy, roll_deg, pitch_deg, yaw_deg):
    """Convert pixel detection to NED unit bearing vector."""
    bx = (cy - IMAGE_H / 2) / FOCAL_LEN
    by = (cx - IMAGE_W / 2) / FOCAL_LEN
    bz = -1.0
    mag = math.sqrt(bx * bx + by * by + bz * bz)
    n, e, d = body_to_ned(bx / mag, by / mag, bz / mag,
                           roll_deg, pitch_deg, yaw_deg)
    mag = math.sqrt(n * n + e * e + d * d)
    return n / mag, e / mag, d / mag


def ned_bearing_to_pixel(bn, be, bd, roll_deg, pitch_deg, yaw_deg):
    """Convert NED bearing back to pixel coords (for visualization)."""
    bx, by, bz = ned_to_body(bn, be, bd, roll_deg, pitch_deg, yaw_deg)
    if bz > -0.01:
        return IMAGE_W // 2, IMAGE_H // 2
    scale = FOCAL_LEN / (-bz)
    px = int(IMAGE_W / 2 + by * scale)
    py = int(IMAGE_H / 2 + bx * scale)
    return px, py


# =========================
# TARGET TRACKER (NED bearing space)
# =========================
class TargetTracker:
    """Tracks target NED bearing and radius using EMA filtering.

    Bearing is tracked in NED frame to eliminate drone-tilt contamination.
    Pixel velocity from drone roll/pitch shifts does NOT affect the estimate.
    """

    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        # NED bearing state
        self.prev_bn = None
        self.prev_be = None
        self.prev_bd = None
        self.prev_time = None
        self.dbn = 0.0
        self.dbe = 0.0
        self.dbd = 0.0
        # Radius (pixel space, for distance estimation)
        self.prev_radius = None
        self.vr_px = 0.0
        # Current values
        self.last_bn = 0.0
        self.last_be = 0.0
        self.last_bd = -1.0
        self.last_radius = 0.0
        self.last_cx = 0.0
        self.last_cy = 0.0
        #
        self.initialized = False
        self.frames_since_update = 0

    def update(self, cx, cy, radius, roll_deg, pitch_deg, yaw_deg, t):
        """Feed new detection with current attitude. Returns True if velocity valid."""
        self.frames_since_update = 0
        self.last_cx = cx
        self.last_cy = cy
        self.last_radius = radius

        bn, be, bd = pixel_to_ned_bearing(cx, cy, roll_deg, pitch_deg, yaw_deg)
        self.last_bn = bn
        self.last_be = be
        self.last_bd = bd

        if self.prev_time is not None:
            dt = t - self.prev_time
            if dt < MIN_DT:
                return self.initialized
            if dt > MAX_DT:
                self.dbn = self.dbe = self.dbd = 0.0
                self.vr_px = 0.0
                self._store(bn, be, bd, radius, t)
                self.initialized = True
                return False

            raw_dbn = (bn - self.prev_bn) / dt
            raw_dbe = (be - self.prev_be) / dt
            raw_dbd = (bd - self.prev_bd) / dt
            raw_vr = (radius - self.prev_radius) / dt

            if self.initialized:
                self.dbn = self.alpha * raw_dbn + (1 - self.alpha) * self.dbn
                self.dbe = self.alpha * raw_dbe + (1 - self.alpha) * self.dbe
                self.dbd = self.alpha * raw_dbd + (1 - self.alpha) * self.dbd
                self.vr_px = self.alpha * raw_vr + (1 - self.alpha) * self.vr_px
            else:
                self.dbn = raw_dbn
                self.dbe = raw_dbe
                self.dbd = raw_dbd
                self.vr_px = raw_vr
                self.initialized = True

        self._store(bn, be, bd, radius, t)
        return self.initialized

    def _store(self, bn, be, bd, radius, t):
        self.prev_bn = bn
        self.prev_be = be
        self.prev_bd = bd
        self.prev_radius = radius
        self.prev_time = t

    def predict_bearing(self, dt_forward):
        """Predict NED bearing dt_forward seconds ahead. Returns unit vector."""
        pn = self.last_bn + self.dbn * dt_forward
        pe = self.last_be + self.dbe * dt_forward
        pd = self.last_bd + self.dbd * dt_forward
        mag = math.sqrt(pn * pn + pe * pe + pd * pd)
        if mag < 1e-6:
            return 0.0, 0.0, -1.0
        return pn / mag, pe / mag, pd / mag

    def predict_radius(self, dt_forward):
        """Predict pixel radius dt_forward seconds ahead."""
        return max(self.last_radius + self.vr_px * dt_forward, 1.0)

    def mark_lost(self):
        self.frames_since_update += 1

    def distance_estimate(self, radius_px=None):
        r = radius_px if radius_px is not None else self.last_radius
        if r < 1.0:
            return 999.0
        return SPHERE_RADIUS * FOCAL_LEN / r

    def visual_closing_speed(self):
        r = self.last_radius
        if r < 1.0:
            return 0.0
        return SPHERE_RADIUS * FOCAL_LEN * self.vr_px / (r * r)


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
drone_yaw_deg = 0.0
drone_roll_deg = 0.0
drone_pitch_deg = 0.0


async def subscribe_telemetry(drone):
    global drone_yaw_deg, drone_roll_deg, drone_pitch_deg

    async for att in drone.telemetry.attitude_euler():
        drone_yaw_deg = att.yaw_deg
        drone_roll_deg = att.roll_deg
        drone_pitch_deg = att.pitch_deg


# =========================
# GUIDANCE
# =========================
def approach_speed(dist):
    """2-zone vertical speed profile. Within MPC_Z_VEL_MAX_UP (3.0)."""
    if dist > APPROACH_FAR_DIST:
        return APPROACH_FAR_SPEED
    if dist > APPROACH_CLOSE_DIST:
        frac = (APPROACH_FAR_DIST - dist) / (APPROACH_FAR_DIST - APPROACH_CLOSE_DIST)
        return APPROACH_FAR_SPEED + frac * (APPROACH_CLOSE_SPEED - APPROACH_FAR_SPEED)
    return APPROACH_CLOSE_SPEED


def compute_guidance(tracker, dist_est, prev_cmd):
    """Compute NED velocity: bearing-dependent vertical + feedforward + proportional lateral.

    Vertical: approach_speed scaled by bearing D component (auto-adjusts to target altitude).
    Lateral: feedforward (match target velocity) + K_LATERAL × bearing_angle (close gap).
    """
    vd_speed = approach_speed(dist_est)

    # Feedforward: estimate target lateral velocity from bearing rate + drone velocity
    ff_vn = prev_cmd[0] + dist_est * tracker.dbn
    ff_ve = prev_cmd[1] + dist_est * tracker.dbe

    # Proportional correction from predicted bearing offset
    t_int = min(dist_est / max(vd_speed, 1.0), MAX_INTERCEPT_TIME)
    pn, pe, pd = tracker.predict_bearing(t_int)

    # Vertical: scale by bearing D (pd<0 = target above = go up, pd>0 = below = descend)
    vd = vd_speed * pd

    lat_mag = math.sqrt(pn * pn + pe * pe)
    if lat_mag > 0.001:
        angle = math.atan2(lat_mag, -pd)
        corr = K_LATERAL * angle
        vn = ff_vn + corr * (pn / lat_mag)
        ve = ff_ve + corr * (pe / lat_mag)
    else:
        vn, ve = ff_vn, ff_ve

    # Cap total lateral velocity
    lat = math.sqrt(vn * vn + ve * ve)
    if lat > MAX_LATERAL_VEL:
        vn *= MAX_LATERAL_VEL / lat
        ve *= MAX_LATERAL_VEL / lat

    info = {
        'dist': dist_est, 'vd_speed': vd_speed, 't_int': t_int,
        'bn': pn, 'be': pe, 'bd': pd,
        'lat_angle': math.degrees(angle) if lat_mag > 0.001 else 0.0,
        'ff': math.sqrt(ff_vn ** 2 + ff_ve ** 2),
    }

    return vn, ve, vd, info


def smooth_cmd(vn, ve, vd, cmd, alpha=CMD_SMOOTH):
    """EMA-smooth velocity commands. Mutates cmd list [vn, ve, vd]."""
    cmd[0] = alpha * vn + (1 - alpha) * cmd[0]
    cmd[1] = alpha * ve + (1 - alpha) * cmd[1]
    cmd[2] = alpha * vd + (1 - alpha) * cmd[2]
    return cmd[0], cmd[1], cmd[2]


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

    telemetry_task = asyncio.create_task(subscribe_telemetry(drone))

    print("Waiting for EKF init...")
    await asyncio.sleep(10)

    print("Priming offboard...")
    for _ in range(20):
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
        await asyncio.sleep(0.1)

    print("Starting offboard...")
    await drone.offboard.start()

    print("Arming...")
    for attempt in range(10):
        try:
            await drone.action.arm()
            print("Armed")
            break
        except Exception as e:
            print(f"Arm attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2)

    print("Taking off...")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, -TAKEOFF_SPEED, 0))
    await asyncio.sleep(TAKEOFF_TIME)

    cap = cv2.VideoCapture(
        "udpsrc port=5600 ! application/x-rtp,encoding-name=H264,payload=96 ! "
        "rtph264depay ! decodebin ! videoconvert ! appsink sync=false",
        cv2.CAP_GSTREAMER
    )

    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

    tracker = TargetTracker()
    search_time = 0.0
    cmd = [0.0, 0.0, 0.0]

    try:
        while True:
            frame_time = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                continue

            roll, pitch, yaw = drone_roll_deg, drone_pitch_deg, drone_yaw_deg
            result = detect_red_object(frame)

            if result:
                cx, cy, radius_px = result
                tracker.update(cx, cy, radius_px, roll, pitch, yaw, frame_time)
                search_time = 0.0

                dist_est = tracker.distance_estimate(radius_px)
                vn, ve, vd, info = compute_guidance(tracker, dist_est, cmd)

                sn, se, sd = smooth_cmd(vn, ve, vd, cmd)
                await drone.offboard.set_velocity_ned(
                    VelocityNedYaw(sn, se, sd, 0.0)
                )

                # Visualization
                cv2.circle(frame, (cx, cy), int(radius_px), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                cv2.circle(frame, (IMAGE_W // 2, IMAGE_H // 2), 4, (0, 0, 255), -1)
                # Show predicted bearing in image
                pcx, pcy = ned_bearing_to_pixel(
                    info['bn'], info['be'], info['bd'], roll, pitch, yaw
                )
                cv2.circle(frame, (pcx, pcy), 6, (0, 255, 255), 2)
                cv2.line(frame, (IMAGE_W // 2, IMAGE_H // 2), (pcx, pcy),
                         (0, 255, 255), 1)

                cv2.putText(frame,
                    f"d:{dist_est:.1f}m vd:{info['vd_speed']:.1f} "
                    f"v:({sn:.1f},{se:.1f},{sd:.1f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame,
                    f"lat:{info['lat_angle']:.1f}deg ff:{info['ff']:.1f} "
                    f"Tint:{info['t_int']:.2f}s",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if dist_est < HIT_DISTANCE:
                    cv2.putText(frame, "HIT!", (IMAGE_W // 2 - 50, IMAGE_H // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
                    cv2.imshow("Tracking", frame)
                    cv2.waitKey(3000)
                    break

            else:
                tracker.mark_lost()
                frames_lost = tracker.frames_since_update

                if frames_lost <= LOST_PREDICT_FRAMES and tracker.initialized:
                    dt_since = frames_lost / 30.0
                    pred_r = tracker.predict_radius(dt_since)
                    dist_est = tracker.distance_estimate(pred_r)
                    vn, ve, vd, _ = compute_guidance(tracker, dist_est, cmd)
                    sn, se, sd = smooth_cmd(
                        vn * 0.5, ve * 0.5, vd * 0.5, cmd
                    )
                    await drone.offboard.set_velocity_ned(
                        VelocityNedYaw(sn, se, sd, 0.0)
                    )
                    cv2.putText(frame, "COASTING...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                elif frames_lost <= LOST_DECEL_FRAMES and tracker.initialized:
                    confidence = 1.0 - (
                        (frames_lost - LOST_PREDICT_FRAMES)
                        / (LOST_DECEL_FRAMES - LOST_PREDICT_FRAMES)
                    )
                    dt_since = frames_lost / 30.0
                    pred_r = tracker.predict_radius(dt_since)
                    dist_est = tracker.distance_estimate(pred_r)
                    vn, ve, vd, _ = compute_guidance(tracker, dist_est, cmd)
                    fade = 0.3 * confidence
                    sn, se, sd = smooth_cmd(
                        vn * fade, ve * fade, vd * fade, cmd
                    )
                    await drone.offboard.set_velocity_ned(
                        VelocityNedYaw(sn, se, sd, 0.0)
                    )
                    cv2.putText(frame, f"DECELERATING... {confidence:.0%}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 165, 255), 2)

                else:
                    search_time += 1.0 / 30.0
                    bias_dir = (
                        math.atan2(tracker.dbe, tracker.dbn)
                        if tracker.initialized else 0.0
                    )
                    angle = SEARCH_OMEGA * search_time + bias_dir
                    vn = SEARCH_LATERAL_SPEED * math.cos(angle)
                    ve = SEARCH_LATERAL_SPEED * math.sin(angle)
                    vd = 0.0  # hold altitude — don't overshoot further

                    sn, se, sd = smooth_cmd(vn, ve, vd, cmd)
                    await drone.offboard.set_velocity_ned(
                        VelocityNedYaw(sn, se, sd, 0.0)
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
