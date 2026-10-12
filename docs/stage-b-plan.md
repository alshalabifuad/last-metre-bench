# Stage B plan — ROS 2 Lyrical + Gazebo Jetty (target: November 2026)

Goal: the same mission as Stage A, running in Gazebo with a mobile base and a placer, visualised in RViz, recorded as a 60-second video for the progress e-mail. The Python core (`lastmetre/`) is reused **unchanged**; ROS 2 nodes are thin wrappers.

## 0. Environment (once)
Ubuntu 26.04 in WSL2 → ROS 2 Lyrical → `ros-lyrical-ros-gz` (Gazebo Jetty). Commands: `uxo-demining-robotics-kfupm/07-demo-spec.md` §0. Check `glxinfo -B` is not `llvmpipe`.

## 1. Packages (colcon workspace `~/lm_ws/src`)
| Package | Type | Content |
|---|---|---|
| `lastmetre` | pure Python | this repo's core (symlink or `pip install -e`) |
| `lm_description` | ament_cmake | URDF/xacro: box base (circular footprint 0.3 m), diff-drive, `sensor_bar` link 1.2 m ahead, 2-DoF placer (yaw + prismatic) with `ee` link; `gz_ros2_control/GazeboSimSystem` |
| `lm_gazebo` | ament_cmake | `field.sdf` (flat 16 × 16 m), surrogate models spawned from `results/field_seed*.json` (visual only, no collision), bridge YAML (`/clock`, `/tf`, `/odom`, `/cmd_vel` TwistStamped, joint states), launch `sim.launch.py` |
| `lm_nodes` | ament_python | `sensor_node` (TF → sensor-bar samples → simulated signal from the field JSON, publishes `lm_msgs`-free `Float32MultiArray` + `PointCloud2` for RViz), `belief_node` (HazardBelief → `nav_msgs/OccupancyGrid` `/hazard_map`, `/risk_map`), `safety_filter_node` (`/cmd_vel_nom` → CBF → `/cmd_vel`, publishes `h`), `mission_node` (survey + chain state machine; operator approvals via a ROS 2 **service** `~/approve` and initiation via `~/operator_initiate` that only a human GUI/CLI calls), `eval_node` (virtual triggers, metrics CSV) |
| `lm_bringup` | ament_python | one launch for everything + RViz config |

## 2. Milestones
1. Robot spawns in Gazebo Jetty; teleop with `teleop_twist_keyboard --ros-args -p stamped:=true`; TF tree OK.
2. `sensor_node` + `belief_node`: driving by hand paints `/hazard_map` in RViz.
3. `safety_filter_node`: driving by hand **cannot** enter unsensed ground (record this clip — it is the core idea in 10 s).
4. `mission_node` survey; then the chain with the placer; operator approval/initiation via service calls from a terminal (record).
5. `eval_node` metrics match Stage A within noise on seeds 1-3.

## 3. Lyrical / Jetty gotchas (from the 11 Oct 2026 stack check)
- `gz sim`, `ros2 run ros_gz_sim create -topic robot_description`, `GZ_SIM_RESOURCE_PATH`.
- diff-drive and Nav2 use **TwistStamped** on `cmd_vel`.
- Controller manager reads `robot_description` from the topic; controller params via spawner `--param-file`.
- `publish_dynamic_joint_states` now defaults to false; JointGroup*Controllers deprecated → `forward_command_controller` with `interface_name`.
- Python 3.14; do not pip-upgrade setuptools.
- RViz cannot draw URDF 1.2 capsules — use cylinders.
