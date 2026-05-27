# cyberdog_develop
开发机器狗
### 1.1 启动仿真

容器内执行：

启动 Gazebo：
```
cd /home/cyberdog_sim
bash src/cyberdog_simulator/cyberdog_gazebo/script/launchgazebo.sh
```
带激光雷达启动：
```
cd /home/cyberdog_sim
bash src/cyberdog_simulator/cyberdog_gazebo/script/launchgazebo_lidar.sh
```
启动控制器：
```
cd /home/cyberdog_sim
bash src/cyberdog_simulator/cyberdog_gazebo/script/launchcontrol.sh
```

需要可视化时再启动rviz：
```
cd /home/cyberdog_sim
bash src/cyberdog_simulator/cyberdog_gazebo/script/launchvisual.sh
```

每次用终端运行脚本前source环境：
```
source /opt/ros/galactic/setup.bash
source /home/cyberdog_ws/install/setup.bash
source /home/cyberdog_sim/install/setup.bash
```
启动相机画面：
```bash
export QT_X11_NO_MITSHM=1
export GDK_DISABLE_SHM=1

ros2 run image_tools showimage --ros-args \
  -p reliability:=best_effort \
  --remap image:=/rgb_camera/rgb_camera_sensor/image_raw
```
报告狗的位置：
```bash
ros2 service call /gazebo/get_entity_state gazebo_msgs/srv/GetEntityState "{name: 'robot', reference_frame: 'world'}"
```
设置狗的位置：
```bash
ros2 service call /gazebo/set_entity_state gazebo_msgs/srv/SetEntityState "{state: {name: 'robot', pose: {position: {x: 2.751, y: 0.985, z: 0.25}, orientation: {x: 0.0, y: 0.0, z: 0.4778, w: 0.8785}}, twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}, reference_frame: 'world'}}"


