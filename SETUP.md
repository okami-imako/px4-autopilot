# Gazebo with PX4 sitl setup

## Gazebo

### Linux

```
docker compose build
xhost +local:root
docker compose up -d
docker exec -it px4-gz bash
```

inside container:
```
make distclean
make px4_sitl gz_x500_mono_cam
```

### Mac (Apple Silicon)

X11 forwarding doesn't support OpenGL on Mac, so the setup uses Xvfb (virtual framebuffer) inside the container with VNC to view the UI.

#### Prerequisites (one time)

Install a VNC client — Mac's built-in Screen Sharing works:
```
brew install --cask tiger-vnc
```
Or just use Finder → Go → Connect to Server.

#### Build

Build the base image first, then the Mac image on top of it:
```
docker compose build
docker compose -f docker-compose.mac.yml build
```

#### Run

```
docker compose -f docker-compose.mac.yml up -d
docker exec -it px4-gz bash
```

#### Connect to the display

Open VNC from your Mac (password: `px4`):
```
open vnc://localhost:5900
```

You will see a black screen — this is normal. The Gazebo window will appear after the sim starts.

#### Start the sim

Inside the container (first run only, clears stale CMake cache):
```
make distclean
make px4_sitl gz_x500_mono_cam
```

Subsequent runs:
```
make px4_sitl gz_x500_mono_cam
```

#### Consume drone camera feed
```
gst-launch-1.0 udpsrc port=5600 \
    ! application/x-rtp,encoding-name=H264,payload=96 \
    ! rtph264depay \
    ! avdec_h264 \
    ! videoconvert \
    ! autovideosink
```

### Camera feed (Linux)
```
gst-launch-1.0 udpsrc port=5600 \
    ! application/x-rtp,encoding-name=H264,payload=96 \
    ! rtph264depay \
    ! avdec_h264 \
    ! videoconvert \
    ! autovideosink
```

## Gazebo Classic
### Step 1 - Clone PX4-Autopilot
```
git clone https://github.com/okami-imako/PX4-Autopilot.git
cd PX4-Autopilot
git submodule update --init --recursive
```

### Step 2 - Container prerequisites
Run the following container (from PX4-Autopilot directory for mounting via pwd):

```
docker run -it --rm --privileged --gpus all\
  --network host \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/src/PX4-Autopilot \
  px4io/px4-dev-simulation-focal
```

> to run in headless mode add `-e HEADLESS=1` (haven't tested this)

Then inside container

```
cd /src/PX4-Autopilot
git config --global --add safe.directory /src/PX4-Autopilot
```

### Step 3 - Running the sim

To run the sim you need to start the container (if you don't already have a running instance):
```
docker run -it --rm --privileged --gpus all\
  --network host \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/src/PX4-Autopilot:rw \
  px4io/px4-dev-simulation-focal
```

And then run the following

```
cd /src/PX4-Autopilot
make px4_sitl gazebo-classic_iris_fpv_cam
```

### To consume video stream
```
gst-launch-1.0 udpsrc port=5600 \
    ! application/x-rtp,encoding-name=H264,payload=96 \
    ! rtph264depay \
    ! avdec_h264 \
    ! videoconvert \
    ! autovideosink
```

## Step 4 - Overcoming arming with disabled GPS
We disabled gps and connected safeguards on PX4 level
Trying to fly the drone will still fail
Running QGroundControl along with the sim helps for some reason

This should open gazebo with loaded drone model

Video feed is available at udp port 5600
Mavlink commands are sent to udp port 14540

Python script `/scripts/detect_and_run.py` finds the sphere and attempts to fly towards it
