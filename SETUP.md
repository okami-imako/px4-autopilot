# Gazebo with PX4 sitl setup

## Step 1 - Clone PX4-Autopilot
```
git clone https://github.com/PX4/PX4-Autopilot.git
cd PX4-Autopilot
git submodule update --init --recursive
```

## Step 2 - Container prerequisites
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

## Step 3 - Running the sim

To run the sim you need to start the container (if you don't already have a running instance):
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

And then run the following

```
cd /src/PX4-Autopilot
make px4_sitl gazebo-classic_typhoon_h480
```

This should open gazebo with loaded drone model

Video feed is available at udp port 5600
Mavlink commands are sent to udp port 14540

# Modifying drone camera position and angle
Go to the drone model

```
PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/typhoon_h480/typhoon_h480.sdf
```

1) Find `<sensor name="camera" type="camera">` entry
2) modify the `<pose>` in order to put it on the top and aim up
   I used `<pose>0.0 0 0.162 0 -1.5708 0</pose>`
3) Adjust `<horizontal_fov>` inside `<sensor><camera>` (in radians)
4) Adjust image resolution `<width>` and `<height>` inside `<sensor><camera><image>`
5) Adjust `<update_rate>` to desired fps inside `<sensor>`

# Adding Red Sphere
## Creating a model

```
mkdir ./Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/red_sphere
touch ./Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/red_sphere/model.sdf
touch ./Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/red_sphere/model.config
```

In `model.sdf` put:

```
<?xml version="2.0" ?>

<sdf version="1.6">
  <model name="red_sphere">

    <pose>0 0 0 0 0 0</pose>

    <link name="link">

      <static>false</static>
      <gravity>false</gravity>
      <!-- <kinematic>true</kinematic> -->

      <inertial>
        <mass>0.1</mass>
      </inertial>

      <visual name="visual">
        <geometry>
          <sphere>
            <radius>0.2</radius>
          </sphere>
        </geometry>

        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
        </material>
      </visual>

      <collision name="collision">
        <geometry>
          <sphere>
            <radius>0.2</radius>
          </sphere>
        </geometry>
      </collision>

    </link>


    <!-- this plugin moves the sphere -->
    <plugin name="red_sphere_move"
          filename="libRedSphereMovePlugin.so"/>

  </model>
</sdf>
```

In model.config:
```
<?xml version="1.0"?>

<model>
  <name>red_sphere</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>

  <author>
    <name>you</name>
    <email>none</email>
  </author>

  <description>
    Moving red sphere test object for PX4 Gazebo simulation
  </description>
</model>
```

## Plugin that moves the sphere

Let's create a plugin that moves the sphere

```
touch ./Tools/simulation/gazebo-classic/sitl_gazebo-classic/src/RedSphereMovePlugin.cpp
```

Put this code inside:
```
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>

#include <sdf/sdf.hh>

namespace gazebo
{
class RedSphereMovePlugin : public ModelPlugin
{
public:
  void Load(physics::ModelPtr _model, sdf::ElementPtr /*_sdf*/)
  {
    this->model = _model;
    this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&RedSphereMovePlugin::OnUpdate, this));

    this->startTime = this->model->GetWorld()->SimTime().Double();

    gzmsg << "[RedSphereMovePlugin] Loaded\n";
  }

  void OnUpdate()
  {
    double t = this->model->GetWorld()->SimTime().Double() - this->startTime;

    double x = 2.0 * sin(t);   // horizontal motion
    double y = 0.0;
    double z = 10.0;            // height above ground

    ignition::math::Pose3d pose(
        ignition::math::Vector3d(x, y, z),
        ignition::math::Quaterniond(0, 0, 0));

    this->model->SetWorldPose(pose);
  }

private:
  physics::ModelPtr model;
  event::ConnectionPtr updateConnection;
  double startTime;
};

GZ_REGISTER_MODEL_PLUGIN(RedSphereMovePlugin)
}
```

Now add the plugin to `./Tools/simulation/gazebo-classic/sitl_gazebo-classic/CMakeLists.txt`
Insert this just before `include(CPack)` in the end of the file:

```
add_library(RedSphereMovePlugin SHARED
  src/RedSphereMovePlugin.cpp
)

target_link_libraries(RedSphereMovePlugin
  ${GAZEBO_LIBRARIES}
)
```

## Adding sphere to the world
The only thing left is to add the sphere to the world
Go to `./Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds/typhoon_h480.world`
And add the following entry

```
<include>
  <uri>model://red_sphere</uri>
  <pose>0 0 50 0 0 0</pose>
</include>
```

Note that the plugin that moves the sphere hard sets the pose

So if you want for example the sphere to be lower or higher you need to modify the `z` in the `RedSphereMovePlugin.cpp`

Now when you run the sim with `make px4_sitl gazebo-classic_typhoon_h480` you should have the sphere in the sky
