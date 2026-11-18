import json

import numpy as np


def make_dataset(root):
    from rosbags.rosbag2 import Writer
    from rosbags.rosbag2.enums import StoragePlugin
    from rosbags.typesys import Stores,get_typestore
    store=get_typestore(Stores.ROS2_HUMBLE);t=store.types;bag=root/'bag';root.mkdir()
    Time=t['builtin_interfaces/msg/Time'];Header=t['std_msgs/msg/Header'];Point=t['geometry_msgs/msg/Point'];Quat=t['geometry_msgs/msg/Quaternion'];Vec=t['geometry_msgs/msg/Vector3'];Pose=t['geometry_msgs/msg/Pose'];Cov=t['geometry_msgs/msg/PoseWithCovariance'];Twist=t['geometry_msgs/msg/Twist'];TCov=t['geometry_msgs/msg/TwistWithCovariance']
    with Writer(bag,version=9,storage_plugin=StoragePlugin.MCAP) as writer:
        image=writer.add_connection('/aima/hal/camera/front/image_raw','sensor_msgs/msg/Image',typestore=store);lidar=writer.add_connection('/aima/hal/lidar/front/point_cloud','sensor_msgs/msg/PointCloud2',typestore=store);odom=writer.add_connection('/map_odometry','nav_msgs/msg/Odometry',typestore=store)
        for i in range(3):
            header=Header(Time(1,i*100000000),'camera');m=t['sensor_msgs/msg/Image'](header,8,16,'nv12',0,16,np.full(192,128,np.uint8));writer.write(image,10**18+i*10**8,store.serialize_cdr(m,'sensor_msgs/msg/Image'))
            fields=[t['sensor_msgs/msg/PointField'](name,k*4,7,1) for k,name in enumerate(['x','y','z'])];xyz=np.array([[0,0,1],[.15,0,1],[0,.15,1],[.15,.15,1]],dtype='<f4');m=t['sensor_msgs/msg/PointCloud2'](Header(Time(1,i*100000000),'rslidar_front'),1,4,fields,False,12,48,xyz.view(np.uint8).reshape(-1),True);writer.write(lidar,10**18+i*10**8+1,store.serialize_cdr(m,'sensor_msgs/msg/PointCloud2'))
            m=t['nav_msgs/msg/Odometry'](Header(Time(1,i*100000000),'world'),'base_link',Cov(Pose(Point(0.,0.,.5),Quat(0.,0.,0.,1.)),np.zeros(36)),TCov(Twist(Vec(0.,0.,0.),Vec(0.,0.,0.)),np.zeros(36)));writer.write(odom,10**18+i*10**8+2,store.serialize_cdr(m,'nav_msgs/msg/Odometry'))
    calibration=[{'source_frame':'camera','target_frame':'base_link','transform':np.eye(4).tolist(),'camera_matrix':[[12,0,7.5],[0,12,3.5],[0,0,1]],'distortion_coefficients':[0.]*8,'image_size':[16,8]},{'source_frame':'rslidar_front','target_frame':'base_link','transform':np.eye(4).tolist()}]
    (root/'extrinst.txt').write_text('\n'.join(json.dumps(v) for v in calibration));return root

