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

# Lateral control (PD via lead pursuit)
KP_CENTER = 0.5              # m/s per normalized pixel error
MAX_LATERAL_VEL = 5.0        # m/s max lateral command
CMD_SMOOTH = 0.3             # EMA alpha for velocity command smoothing (lower = smoother)

# Approach speed profile (4 zones by distance)
APPROACH_FAR_DIST = 12.0     # meters — far/medium boundary
APPROACH_MED_DIST = 5.0      # meters — medium/close boundary
APPROACH_CLOSE_DIST = 2.0    # meters — close/terminal boundary
APPROACH_FAR_SPEED = 3.0     # m/s upward in far zone
APPROACH_MED_MAX = 7.0       # m/s upward at close end of medium zone
APPROACH_CLOSE_SPEED = 8.0   # m/s upward in close zone
APPROACH_TERMINAL_SPEED = 10.0  # m/s max lunge
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
SEARCH_ASCEND_SPEED = 1.0    # m/s upward (target is above)
SEARCH_LATERAL_SPEED = 1.5   # m/s lateral during search
SEARCH_OMEGA = 0.8           # rad/s angular speed of circle
LOST_PREDICT_FRAMES = 10     # coast on prediction
LOST_DECEL_FRAMES = 30       # decelerate, fade prediction
LOST_SEARCH_FRAMES = 30      # enter full search pattern


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
        # Filtered pixel velocities (pixels/second)
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
def compute_guidance(tracker, dist_est, yaw_deg, roll_deg, pitch_deg):
    """Compute NED velocity command using lead pursuit with drift/tilt compensation.
    Returns (vn, ve, vd, info_dict).
    """
    # Approach speed (4-zone profile)
    if dist_est > APPROACH_FAR_DIST:
        approach_speed = APPROACH_FAR_SPEED
    elif dist_est > APPROACH_MED_DIST:
        frac = (APPROACH_FAR_DIST - dist_est) / (APPROACH_FAR_DIST - APPROACH_MED_DIST)
        approach_speed = APPROACH_FAR_SPEED + frac * (APPROACH_MED_MAX - APPROACH_FAR_SPEED)
    elif dist_est > APPROACH_CLOSE_DIST:
        approach_speed = APPROACH_CLOSE_SPEED
    else:
        approach_speed = APPROACH_TERMINAL_SPEED

    # Drift compensation: if visual closing speed much lower than commanded, boost
    vis_cs = tracker.visual_closing_speed()
    if tracker.initialized and vis_cs < 0.5 * approach_speed:
        approach_speed *= DRIFT_BOOST

    vd = -approach_speed  # NED: negative = up

    # Intercept time
    closing_speed = max(approach_speed, 1.0)
    t_intercept = min(dist_est / closing_speed, MAX_INTERCEPT_TIME)

    # Predicted target pixel position
    pred_cx, pred_cy, _ = tracker.predict(t_intercept)

    # Tilt compensation: remove false centering caused by drone inclination
    # Roll right → target shifted left in image → add correction to X
    # Pitch up → target shifted down in image → subtract correction from Y
    roll_rad = math.radians(roll_deg)
    pitch_rad = math.radians(pitch_deg)
    pred_cx += roll_rad * FOCAL_LEN
    pred_cy -= pitch_rad * FOCAL_LEN

    # Normalized pixel error toward corrected position [-1, 1] (clamped to [-1.5, 1.5])
    ex = (pred_cx - IMAGE_W / 2) / (IMAGE_W / 2)
    ey = (IMAGE_H / 2 - pred_cy) / (IMAGE_H / 2)
    ex = max(-1.5, min(1.5, ex))
    ey = max(-1.5, min(1.5, ey))

    # Yaw-compensated pixel-to-NED
    yaw_rad = math.radians(yaw_deg)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    vn = KP_CENTER * (cos_y * ey - sin_y * ex)
    ve = KP_CENTER * (sin_y * ey + cos_y * ex)

    # Clamp lateral velocity
    mag = math.sqrt(vn**2 + ve**2)
    if mag > MAX_LATERAL_VEL:
        scale = MAX_LATERAL_VEL / mag
        vn *= scale
        ve *= scale

    info = {
        'dist': dist_est,
        'approach': approach_speed,
        't_int': t_intercept,
        'ex': ex, 'ey': ey,
        'vx_px': tracker.vx_px,
        'vy_px': tracker.vy_px,
        'vis_cs': vis_cs,
        'pred_cx': pred_cx, 'pred_cy': pred_cy,
    }

    return vn, ve, vd, info


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
    cmd_vn, cmd_ve, cmd_vd = 0.0, 0.0, 0.0  # smoothed velocity commands

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
                vn, ve, vd, info = compute_guidance(tracker, dist_est, drone_yaw_deg, drone_roll_deg, drone_pitch_deg)

                # Smooth velocity commands to prevent jerky direction changes
                a = CMD_SMOOTH
                cmd_vn = a * vn + (1 - a) * cmd_vn
                cmd_ve = a * ve + (1 - a) * cmd_ve
                cmd_vd = a * vd + (1 - a) * cmd_vd

                await drone.offboard.set_velocity_ned(
                    VelocityNedYaw(cmd_vn, cmd_ve, cmd_vd, 0.0)
                )

                # Visualization
                cv2.circle(frame, (cx, cy), int(radius_px), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                cv2.circle(frame, (IMAGE_W // 2, IMAGE_H // 2), 4, (0, 0, 255), -1)
                # Predicted intercept position
                pcx, pcy = int(info['pred_cx']), int(info['pred_cy'])
                cv2.circle(frame, (pcx, pcy), 6, (0, 255, 255), 2)
                cv2.line(frame, (IMAGE_W // 2, IMAGE_H // 2), (pcx, pcy), (0, 255, 255), 1)

                cv2.putText(frame,
                    f"d:{dist_est:.1f}m v:({vn:.1f},{ve:.1f},{vd:.1f}) Tint:{info['t_int']:.2f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame,
                    f"vpx:{info['vx_px']:.0f} vpy:{info['vy_px']:.0f} vcs:{info['vis_cs']:.1f}",
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
                    # Phase 1: coast on prediction
                    dt_since = frames_lost / 30.0
                    _, _, pred_r = tracker.predict(dt_since)
                    dist_est = tracker.distance_estimate(pred_r)
                    vn, ve, vd, info = compute_guidance(tracker, dist_est, drone_yaw_deg, drone_roll_deg, drone_pitch_deg)
                    vd *= 0.5

                    a = CMD_SMOOTH
                    cmd_vn = a * vn + (1 - a) * cmd_vn
                    cmd_ve = a * ve + (1 - a) * cmd_ve
                    cmd_vd = a * vd + (1 - a) * cmd_vd

                    await drone.offboard.set_velocity_ned(
                        VelocityNedYaw(cmd_vn, cmd_ve, cmd_vd, 0.0)
                    )
                    cv2.putText(frame, "COASTING...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                elif frames_lost <= LOST_DECEL_FRAMES and tracker.initialized:
                    # Phase 2: decelerate, fade prediction
                    confidence = 1.0 - (frames_lost - LOST_PREDICT_FRAMES) / (LOST_DECEL_FRAMES - LOST_PREDICT_FRAMES)
                    dt_since = frames_lost / 30.0
                    _, _, pred_r = tracker.predict(dt_since)
                    dist_est = tracker.distance_estimate(pred_r)
                    vn, ve, vd, info = compute_guidance(tracker, dist_est, drone_yaw_deg, drone_roll_deg, drone_pitch_deg)
                    vn *= confidence
                    ve *= confidence
                    vd *= 0.3 * confidence

                    a = CMD_SMOOTH
                    cmd_vn = a * vn + (1 - a) * cmd_vn
                    cmd_ve = a * ve + (1 - a) * cmd_ve
                    cmd_vd = a * vd + (1 - a) * cmd_vd

                    await drone.offboard.set_velocity_ned(
                        VelocityNedYaw(cmd_vn, cmd_ve, cmd_vd, 0.0)
                    )
                    cv2.putText(frame, f"DECELERATING... {confidence:.0%}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                else:
                    # Phase 3: search — ascend + circle biased toward last known velocity
                    search_time += 1.0 / 30.0
                    bias_dir = math.atan2(tracker.vx_px, -tracker.vy_px) if tracker.initialized else 0.0
                    angle = SEARCH_OMEGA * search_time + bias_dir
                    vn = SEARCH_LATERAL_SPEED * math.cos(angle)
                    ve = SEARCH_LATERAL_SPEED * math.sin(angle)
                    vd = -SEARCH_ASCEND_SPEED  # ascend — target is above

                    a = CMD_SMOOTH
                    cmd_vn = a * vn + (1 - a) * cmd_vn
                    cmd_ve = a * ve + (1 - a) * cmd_ve
                    cmd_vd = a * vd + (1 - a) * cmd_vd

                    await drone.offboard.set_velocity_ned(
                        VelocityNedYaw(cmd_vn, cmd_ve, cmd_vd, 0.0)
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
