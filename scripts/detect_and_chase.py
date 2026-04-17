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
CMD_SMOOTH = 0.3             # EMA alpha (lower = smoother)

# Approach speed profile (4 zones) — total 3D speed toward target
APPROACH_FAR_DIST = 15.0     # meters — far/medium boundary
APPROACH_MED_DIST = 5.0      # meters — medium/close boundary
APPROACH_CLOSE_DIST = 2.0    # meters — close/terminal boundary
APPROACH_FAR_SPEED = 5.0     # m/s in far zone
APPROACH_MED_MAX = 10.0      # m/s at close end of medium zone
APPROACH_CLOSE_SPEED = 12.0  # m/s in close zone
APPROACH_TERMINAL_SPEED = 12.0  # m/s max lunge
DRIFT_BOOST = 1.3            # speed multiplier when visual closing speed is low

# Tracker (EMA velocity estimation)
EMA_ALPHA = 0.3              # smoothing factor
MAX_INTERCEPT_TIME = 2.0     # seconds — clamp prediction horizon
MIN_DT = 0.005               # ignore frames closer than 5ms
MAX_DT = 0.2                 # reset velocity if gap exceeds 200ms

# Takeoff
TAKEOFF_SPEED = 2.0          # m/s upward
TAKEOFF_TIME = 5.0           # seconds to climb

# Search (when target lost)
SEARCH_ASCEND_SPEED = 1.0    # m/s upward
SEARCH_LATERAL_SPEED = 1.5   # m/s lateral during search
SEARCH_OMEGA = 0.8           # rad/s angular speed of circle
LOST_PREDICT_FRAMES = 10     # coast on prediction
LOST_DECEL_FRAMES = 30       # decelerate, fade prediction


# =========================
# TARGET TRACKER
# =========================
class TargetTracker:
    """Tracks target position and velocity in pixel space using EMA filtering."""

    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.prev_cx = None
        self.prev_cy = None
        self.prev_radius = None
        self.prev_time = None
        self.vx_px = 0.0
        self.vy_px = 0.0
        self.vr_px = 0.0
        self.initialized = False
        self.frames_since_update = 0
        self.last_cx = 0.0
        self.last_cy = 0.0
        self.last_radius = 0.0

    def update(self, cx, cy, radius, t):
        """Feed new detection. Returns True if velocity estimate is valid."""
        self.frames_since_update = 0
        self.last_cx = cx
        self.last_cy = cy
        self.last_radius = radius

        if self.prev_time is not None:
            dt = t - self.prev_time
            if dt < MIN_DT:
                return self.initialized
            if dt > MAX_DT:
                self.vx_px = 0.0
                self.vy_px = 0.0
                self.vr_px = 0.0
                self._store(cx, cy, radius, t)
                self.initialized = True
                return False

            raw_vx = (cx - self.prev_cx) / dt
            raw_vy = (cy - self.prev_cy) / dt
            raw_vr = (radius - self.prev_radius) / dt

            if self.initialized:
                self.vx_px = self.alpha * raw_vx + (1 - self.alpha) * self.vx_px
                self.vy_px = self.alpha * raw_vy + (1 - self.alpha) * self.vy_px
                self.vr_px = self.alpha * raw_vr + (1 - self.alpha) * self.vr_px
            else:
                self.vx_px = raw_vx
                self.vy_px = raw_vy
                self.vr_px = raw_vr
                self.initialized = True

        self._store(cx, cy, radius, t)
        return self.initialized

    def _store(self, cx, cy, radius, t):
        self.prev_cx = cx
        self.prev_cy = cy
        self.prev_radius = radius
        self.prev_time = t

    def predict(self, dt_forward):
        """Predict target pixel position dt_forward seconds ahead."""
        px = self.last_cx + self.vx_px * dt_forward
        py = self.last_cy + self.vy_px * dt_forward
        pr = max(self.last_radius + self.vr_px * dt_forward, 1.0)
        return px, py, pr

    def mark_lost(self):
        self.frames_since_update += 1

    def distance_estimate(self, radius_px=None):
        r = radius_px if radius_px is not None else self.last_radius
        if r < 1.0:
            return 999.0
        return SPHERE_RADIUS * FOCAL_LEN / r

    def visual_closing_speed(self):
        """Estimate closing speed from radius change rate.
        dist = R*f/r  →  d(dist)/dt = -R*f*vr/r²  →  closing = R*f*vr/r²
        """
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
def body_to_ned(bx, by, bz, roll_deg, pitch_deg, yaw_deg):
    """Rotate body-frame vector to NED using ZYX Euler convention.
    PX4 body: X forward, Y right, Z down.
    NED: X north, Y east, Z down.
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


def approach_speed(dist):
    """4-zone speed profile. Returns total 3D speed toward target."""
    if dist > APPROACH_FAR_DIST:
        return APPROACH_FAR_SPEED
    if dist > APPROACH_MED_DIST:
        frac = (APPROACH_FAR_DIST - dist) / (APPROACH_FAR_DIST - APPROACH_MED_DIST)
        return APPROACH_FAR_SPEED + frac * (APPROACH_MED_MAX - APPROACH_FAR_SPEED)
    if dist > APPROACH_CLOSE_DIST:
        return APPROACH_CLOSE_SPEED
    return APPROACH_TERMINAL_SPEED


def compute_guidance(tracker, dist_est, roll_deg, pitch_deg, yaw_deg):
    """Compute NED velocity using bearing-based 3D pursuit.

    Pixel → body-frame bearing → NED (full rotation matrix) → velocity = speed × direction.
    Tilt compensation is exact (rotation matrix, not approximation).
    Speed naturally splits vertical/lateral based on where target is.
    """
    speed = approach_speed(dist_est)

    # Drift compensation: boost if visual closing speed much lower than commanded
    vis_cs = tracker.visual_closing_speed()
    if tracker.initialized and vis_cs < 0.5 * speed:
        speed *= DRIFT_BOOST

    # Lead pursuit: predict target pixel position at intercept time
    t_int = min(dist_est / max(speed, 1.0), MAX_INTERCEPT_TIME)
    pred_cx, pred_cy, _ = tracker.predict(t_int)

    # Body-frame bearing from predicted pixel position
    # Camera-to-body mapping (empirically confirmed):
    #   image up (lower cy) → body +X (forward)
    #   image right (higher cx) → body +Y (right)
    #   optical axis (up) → body -Z
    bx = (IMAGE_H / 2 - pred_cy) / FOCAL_LEN
    by = (pred_cx - IMAGE_W / 2) / FOCAL_LEN
    bz = -1.0
    mag = math.sqrt(bx * bx + by * by + bz * bz)
    bx /= mag
    by /= mag
    bz /= mag

    # Rotate body bearing to NED using full attitude
    vn, ve, vd = body_to_ned(bx, by, bz, roll_deg, pitch_deg, yaw_deg)

    # Scale by speed
    vn *= speed
    ve *= speed
    vd *= speed

    info = {
        'dist': dist_est, 'speed': speed, 't_int': t_int,
        'bx': bx, 'by': by, 'bz': bz,
        'vx_px': tracker.vx_px, 'vy_px': tracker.vy_px,
        'vis_cs': vis_cs,
        'pred_cx': pred_cx, 'pred_cy': pred_cy,
    }

    return vn, ve, vd, info


def smooth_and_send(vn, ve, vd, cmd, alpha=CMD_SMOOTH):
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
    cmd = [0.0, 0.0, 0.0]  # smoothed [vn, ve, vd]

    try:
        while True:
            frame_time = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                continue

            result = detect_red_object(frame)

            if result:
                cx, cy, radius_px = result
                tracker.update(cx, cy, radius_px, frame_time)
                search_time = 0.0

                dist_est = tracker.distance_estimate(radius_px)
                vn, ve, vd, info = compute_guidance(
                    tracker, dist_est,
                    drone_roll_deg, drone_pitch_deg, drone_yaw_deg
                )

                sn, se, sd = smooth_and_send(vn, ve, vd, cmd)
                await drone.offboard.set_velocity_ned(
                    VelocityNedYaw(sn, se, sd, 0.0)
                )

                # Visualization
                cv2.circle(frame, (cx, cy), int(radius_px), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                cv2.circle(frame, (IMAGE_W // 2, IMAGE_H // 2), 4, (0, 0, 255), -1)
                pcx, pcy = int(info['pred_cx']), int(info['pred_cy'])
                cv2.circle(frame, (pcx, pcy), 6, (0, 255, 255), 2)
                cv2.line(frame, (IMAGE_W // 2, IMAGE_H // 2), (pcx, pcy),
                         (0, 255, 255), 1)

                cv2.putText(frame,
                    f"d:{dist_est:.1f}m spd:{info['speed']:.1f} "
                    f"v:({sn:.1f},{se:.1f},{sd:.1f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame,
                    f"bear:({info['bx']:.2f},{info['by']:.2f},{info['bz']:.2f}) "
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
                    _, _, pred_r = tracker.predict(dt_since)
                    dist_est = tracker.distance_estimate(pred_r)
                    vn, ve, vd, _ = compute_guidance(
                        tracker, dist_est,
                        drone_roll_deg, drone_pitch_deg, drone_yaw_deg
                    )
                    sn, se, sd = smooth_and_send(
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
                    _, _, pred_r = tracker.predict(dt_since)
                    dist_est = tracker.distance_estimate(pred_r)
                    vn, ve, vd, _ = compute_guidance(
                        tracker, dist_est,
                        drone_roll_deg, drone_pitch_deg, drone_yaw_deg
                    )
                    fade = 0.3 * confidence
                    sn, se, sd = smooth_and_send(
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
                        math.atan2(tracker.vx_px, -tracker.vy_px)
                        if tracker.initialized else 0.0
                    )
                    angle = SEARCH_OMEGA * search_time + bias_dir
                    vn = SEARCH_LATERAL_SPEED * math.cos(angle)
                    ve = SEARCH_LATERAL_SPEED * math.sin(angle)
                    vd = -SEARCH_ASCEND_SPEED

                    sn, se, sd = smooth_and_send(vn, ve, vd, cmd)
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
