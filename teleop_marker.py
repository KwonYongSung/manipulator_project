#!/usr/bin/env python
#
# Copyright (c) 2011, Willow Garage, Inc.
# All rights reserved.
#
# Software License Agreement (BSD License 2.0)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of {copyright_holder} nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Will Son

#aruco
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
from ros2_aruco import transformations

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from ros2_aruco_interfaces.msg import ArucoMarkers


#manipulator
from math import exp
import os
import rclpy
import select
import sys
import threading
import time

from open_manipulator_msgs.msg import KinematicsPose, OpenManipulatorState
from open_manipulator_msgs.srv import SetJointPosition, SetKinematicsPose
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from math import radians

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

present_joint_angle = [0.0, 0.0, 0.0, 0.0, 0.0]
goal_joint_angle = [0.0, 0.0, 0.0, 0.0, 0.0]
prev_goal_joint_angle = [0.0, 0.0, 0.0, 0.0, 0.0]
present_kinematics_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
goal_kinematics_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
prev_goal_kinematics_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

task_position_delta = 0.01  # meter
joint_angle_delta = 0.05  # radian
path_time = 0.5  # second

usage = """
Control Your OpenManipulator!
---------------------------
Task Space Control:
         (Forward, X+)
              W                   Q (Upward, Z+)
(Left, Y+) A     D (Right, Y-)    Z (Downward, Z-)
              X 
        (Backward, X-)

Joint Space Control:
- Joint1 : Increase (Y), Decrease (H)
- Joint2 : Increase (U), Decrease (J)
- Joint3 : Increase (I), Decrease (K)
- Joint4 : Increase (O), Decrease (L)
- Gripper: Increase (F), Decrease (G) | Fully Open (V), Fully Close (B)

INIT : (1)
HOME : (2)

CTRL-C to quit
"""

e = """
Communications Failed
"""


class ArucoNode(rclpy.node.Node):

    def __init__(self):
        super().__init__('aruco_node')

        # Declare and read parameters
        self.declare_parameter("marker_size", .07)
        self.declare_parameter("aruco_dictionary_id", "DICT_5X5_250")
        self.declare_parameter("image_topic", "image_raw")
        self.declare_parameter("camera_info_topic", "camera_info")
        #self.declare_parameter("camera_info_topic", "camera/image/camera_info")
        self.declare_parameter("camera_frame", "camera")

        self.marker_size = self.get_parameter("marker_size").get_parameter_value().double_value
        dictionary_id_name = self.get_parameter(
            "aruco_dictionary_id").get_parameter_value().string_value
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value

        # Make sure we have a valid dictionary id:
        try:
            dictionary_id = cv2.aruco.__getattribute__(dictionary_id_name)
            if type(dictionary_id) != type(cv2.aruco.DICT_5X5_100):
                raise AttributeError
        except AttributeError:
            self.get_logger().error("bad aruco_dictionary_id: {}".format(dictionary_id_name))
            options = "\n".join([s for s in dir(cv2.aruco) if s.startswith("DICT")])
            self.get_logger().error("valid options: {}".format(options))

        # Set up subscriptions
        self.info_sub = self.create_subscription(CameraInfo,
                                                 info_topic,
                                                 self.info_callback,
                                                 qos_profile_sensor_data)

        self.create_subscription(Image, image_topic,
                                 self.image_callback, qos_profile_sensor_data)

        # Set up publishers
        self.poses_pub = self.create_publisher(PoseArray, 'aruco_poses', 10)
        self.markers_pub = self.create_publisher(ArucoMarkers, 'aruco_markers', 10)

        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        self.aruco_dictionary = cv2.aruco.Dictionary_get(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        self.bridge = CvBridge()

    def info_callback(self, info_msg):
        self.info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)
        # Assume that camera parameters will remain the same...
        self.destroy_subscription(self.info_sub)

    def image_callback(self, img_msg):
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return
        # cv_image = np.full((320,240), 255, dtype=np.uint8)
        # try:
        # cv_image = self.bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='8UC3')
        # except: 
        #     pass
        
        markers = ArucoMarkers()
        pose_array = PoseArray()
        if self.camera_frame is None:
            markers.header.frame_id = self.info_msg.header.frame_id
            pose_array.header.frame_id = self.info_msg.header.frame_id
        else:
            markers.header.frame_id = self.camera_frame
            pose_array.header.frame_id = self.camera_frame
            
            
        markers.header.stamp = img_msg.header.stamp
        pose_array.header.stamp = img_msg.header.stamp

        corners, marker_ids, rejected = cv2.aruco.detectMarkers(cv_image,
                                                                self.aruco_dictionary,
                                                                parameters=self.aruco_parameters)
        if marker_ids is not None:

            if cv2.__version__ > '4.0.0':
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners,
                                                                      self.marker_size, self.intrinsic_mat,
                                                                      self.distortion)
            else:
                rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners,
                                                                   self.marker_size, self.intrinsic_mat,
                                                                   self.distortion)
            for i, marker_id in enumerate(marker_ids):
                pose = Pose()
                pose.position.x = tvecs[i][0][0]
                pose.position.y = tvecs[i][0][1]
                pose.position.z = tvecs[i][0][2]

                rot_matrix = np.eye(4)
                rot_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
                quat = transformations.quaternion_from_matrix(rot_matrix)

                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]

                pose_array.poses.append(pose)
                markers.poses.append(pose)
                markers.marker_ids.append(marker_id[0])

            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)
            
            try:
                teleop_keyboard = TeleopKeyboard()
            except Exception as e:
                print(e)

            goal_joint_angle[0] = radians(0)
            goal_joint_angle[1] = radians(16)
            goal_joint_angle[2] = 0.00
            goal_joint_angle[3] = -0.26
            pathtime = 2.5
            teleop_keyboard.send_goal_joint_space(pathtime)
            time.sleep(2)
            goal_joint_angle[4] = -0.001  # 물체의 크기에 따라 정도 정하기
            teleop_keyboard.send_tool_control_request()
            time.sleep(1)
            goal_joint_angle[1] = radians(-10)
            pathtime = 3.0
            teleop_keyboard.send_goal_joint_space(pathtime)
            time.sleep(1)
            
            #if marker_ids == [0]:  #0번 마커가 인식되면 특정 위치에 놓기
            goal_joint_angle[0] = radians(180)
            goal_joint_angle[1] = radians(-10)
            goal_joint_angle[2] = 0.00
            goal_joint_angle[3] = 0.00
            pathtime = 3.0
            teleop_keyboard.send_goal_joint_space(pathtime)
            time.sleep(5)
            goal_joint_angle[4] = 0.005
            teleop_keyboard.send_tool_control_request()
            
            
            

class TeleopKeyboard(Node):

    qos = QoSProfile(depth=10)
    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    def __init__(self):
        super().__init__('teleop_keyboard')
        key_value = ''

        # Create joint_states subscriber
        self.joint_state_subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            self.qos)
        self.joint_state_subscription

        # Create kinematics_pose subscriber
        self.kinematics_pose_subscription = self.create_subscription(
            KinematicsPose,
            'kinematics_pose',
            self.kinematics_pose_callback,
            self.qos)
        self.kinematics_pose_subscription

        # Create manipulator state subscriber
        self.open_manipulator_state_subscription = self.create_subscription(
            OpenManipulatorState,
            'states',
            self.open_manipulator_state_callback,
            self.qos)
        self.open_manipulator_state_subscription

        # Create Service Clients
        self.goal_joint_space = self.create_client(SetJointPosition, 'goal_joint_space_path')
        self.goal_task_space = self.create_client(SetKinematicsPose, 'goal_task_space_path')
        self.tool_control = self.create_client(SetJointPosition, 'goal_tool_control')
        self.goal_joint_space_req = SetJointPosition.Request()
        self.goal_task_space_req = SetKinematicsPose.Request()
        self.tool_control_req = SetJointPosition.Request()

    def send_goal_task_space(self):
        self.goal_task_space_req.end_effector_name = 'gripper'
        self.goal_task_space_req.kinematics_pose.pose.position.x = goal_kinematics_pose[0]
        self.goal_task_space_req.kinematics_pose.pose.position.y = goal_kinematics_pose[1]
        self.goal_task_space_req.kinematics_pose.pose.position.z = goal_kinematics_pose[2]
        self.goal_task_space_req.kinematics_pose.pose.orientation.w = goal_kinematics_pose[3]
        self.goal_task_space_req.kinematics_pose.pose.orientation.x = goal_kinematics_pose[4]
        self.goal_task_space_req.kinematics_pose.pose.orientation.y = goal_kinematics_pose[5]
        self.goal_task_space_req.kinematics_pose.pose.orientation.z = goal_kinematics_pose[6]
        self.goal_task_space_req.path_time = path_time

        try:
            self.goal_task_space.call_async(self.goal_task_space_req)
        except Exception as e:
            self.get_logger().info('Sending Goal Kinematic Pose failed %r' % (e,))

    def send_goal_joint_space(self, path_time):
        self.goal_joint_space_req.joint_position.joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'gripper']
        self.goal_joint_space_req.joint_position.position = [goal_joint_angle[0], goal_joint_angle[1], goal_joint_angle[2], goal_joint_angle[3], goal_joint_angle[4]]
        self.goal_joint_space_req.path_time = path_time

        try:
            self.goal_joint_space.call_async(self.goal_joint_space_req)
        except Exception as e:
            self.get_logger().info('Sending Goal Joint failed %r' % (e,))

    def send_tool_control_request(self):
        self.tool_control_req.joint_position.joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'gripper']
        self.tool_control_req.joint_position.position = [goal_joint_angle[0], goal_joint_angle[1], goal_joint_angle[2], goal_joint_angle[3], goal_joint_angle[4]]
        self.tool_control_req.path_time = path_time

        try:
            self.tool_control_result = self.tool_control.call_async(self.tool_control_req)

        except Exception as e:
            self.get_logger().info('Tool control failed %r' % (e,))

    def kinematics_pose_callback(self, msg):
        present_kinematics_pose[0] = msg.pose.position.x
        present_kinematics_pose[1] = msg.pose.position.y
        present_kinematics_pose[2] = msg.pose.position.z
        present_kinematics_pose[3] = msg.pose.orientation.w
        present_kinematics_pose[4] = msg.pose.orientation.x
        present_kinematics_pose[5] = msg.pose.orientation.y
        present_kinematics_pose[6] = msg.pose.orientation.z

    def joint_state_callback(self, msg):
        present_joint_angle[0] = msg.position[0]
        present_joint_angle[1] = msg.position[1]
        present_joint_angle[2] = msg.position[2]
        present_joint_angle[3] = msg.position[3]
        present_joint_angle[4] = msg.position[4]

    def open_manipulator_state_callback(self, msg):
        if msg.open_manipulator_moving_state == 'STOPPED':
            for index in range(0, 7):
                goal_kinematics_pose[index] = present_kinematics_pose[index]
            for index in range(0, 5):
                goal_joint_angle[index] = present_joint_angle[index]

def get_key(settings):
    if os.name == 'nt':
        return msvcrt.getch().decode('utf-8')
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    print_present_values()
    return key

def print_present_values():
    print(usage)
    print('Joint Angle(Rad): [{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}]'.format(
        present_joint_angle[0],
        present_joint_angle[1],
        present_joint_angle[2],
        present_joint_angle[3],
        present_joint_angle[4]))
    print('Kinematics Pose(Pose X, Y, Z | Orientation W, X, Y, Z): {:.3f}, {:.3f}, {:.3f} | {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
        present_kinematics_pose[0],
        present_kinematics_pose[1],
        present_kinematics_pose[2],
        present_kinematics_pose[3],
        present_kinematics_pose[4],
        present_kinematics_pose[5],
        present_kinematics_pose[6]))


def main():

    rclpy.init()
    
    
    try:
        teleop_keyboard = TeleopKeyboard()
    except Exception as e:
        print(e)

    goal_joint_angle[0] = radians(-20)
    goal_joint_angle[1] = -1.4
    goal_joint_angle[2] = 1.1
    goal_joint_angle[3] = 0.35
    goal_joint_angle[4] = 0.01
    pathtime = 5.0
    teleop_keyboard.send_goal_joint_space(pathtime)
    teleop_keyboard.send_tool_control_request()


        
    node = ArucoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

#    try:
#       rclpy.init()
#    except Exception as e:
#        print(e)

    

	#while 1: # 조건 : marker를 찾을 때 까지
		#if goal_joint_angle[0] < radians(30) :
			#goal_joint_angle[0] = prev_goal_joint_angle[0] + joint_angle_delta
		#elif goal_joint_angle[0] > -1.3 :

    try:
        while(rclpy.ok()):
            rclpy.spin_once(teleop_keyboard)
            key_value = get_key(settings)
            if key_value == 'w':
                goal_kinematics_pose[0] = prev_goal_kinematics_pose[0] + task_position_delta
                teleop_keyboard.send_goal_task_space()
            elif key_value == 'x':
                goal_kinematics_pose[0] = prev_goal_kinematics_pose[0] - task_position_delta
                teleop_keyboard.send_goal_task_space()
            elif key_value == 'a':
                goal_kinematics_pose[1] = prev_goal_kinematics_pose[1] + task_position_delta
                teleop_keyboard.send_goal_task_space()
            elif key_value == 'd':
                goal_kinematics_pose[1] = prev_goal_kinematics_pose[1] - task_position_delta
                teleop_keyboard.send_goal_task_space()
            elif key_value == 'q':
                goal_kinematics_pose[2] = prev_goal_kinematics_pose[2] + task_position_delta
                teleop_keyboard.send_goal_task_space()
            elif key_value == 'z':
                goal_kinematics_pose[2] = prev_goal_kinematics_pose[2] - task_position_delta
                teleop_keyboard.send_goal_task_space()
            elif key_value == 'y':
                goal_joint_angle[0] = prev_goal_joint_angle[0] + joint_angle_delta
                pathtime = path_time
                teleop_keyboard.send_goal_joint_space(pathtime)
            elif key_value == 'h':
                goal_joint_angle[0] = prev_goal_joint_angle[0] - joint_angle_delta
                pathtime = path_time
                teleop_keyboard.send_goal_joint_space(pathtime)
            elif key_value == 'u':
                goal_joint_angle[1] = prev_goal_joint_angle[1] + joint_angle_delta
                pathtime = path_time
                teleop_keyboard.send_goal_joint_space(pathtime)
            elif key_value == 'j':
                goal_joint_angle[1] = prev_goal_joint_angle[1] - joint_angle_delta
                pathtime = path_time
                teleop_keyboard.send_goal_joint_space(pathtime)
            elif key_value == 'i':
                goal_joint_angle[2] = prev_goal_joint_angle[2] + joint_angle_delta
                pathtime = path_time
                teleop_keyboard.send_goal_joint_space(pathtime)
            elif key_value == 'k':
                goal_joint_angle[2] = prev_goal_joint_angle[2] - joint_angle_delta
                pathtime = path_time
                teleop_keyboard.send_goal_joint_space(pathtime)
            elif key_value == 'o':
                goal_joint_angle[3] = prev_goal_joint_angle[3] + joint_angle_delta
                pathtime = path_time
                teleop_keyboard.send_goal_joint_space(pathtime)
            elif key_value == 'l':
                goal_joint_angle[3] = prev_goal_joint_angle[3] - joint_angle_delta
                pathtime = path_time
                teleop_keyboard.send_goal_joint_space(pathtime)
            elif key_value == 'f':
                goal_joint_angle[4] = prev_goal_joint_angle[4] + 0.002
                teleop_keyboard.send_tool_control_request()
            elif key_value == 'g':
                goal_joint_angle[4] = prev_goal_joint_angle[4] - 0.002
                teleop_keyboard.send_tool_control_request()
            elif key_value == 'v':
                goal_joint_angle[4] = 0.01
                teleop_keyboard.send_tool_control_request()
            elif key_value == 'b':
                goal_joint_angle[4] = -0.01
                teleop_keyboard.send_tool_control_request()
            elif key_value == '1':
                goal_joint_angle[0] = 0.0
                goal_joint_angle[1] = 0.0
                goal_joint_angle[2] = 0.0
                goal_joint_angle[3] = 0.0
                pathtime = 5.0
                teleop_keyboard.send_goal_joint_space(pathtime)
            elif key_value == '2':
                goal_joint_angle[0] = 0.0
                goal_joint_angle[1] = -1.05
                goal_joint_angle[2] = 0.35
                goal_joint_angle[3] = 0.70
                pathtime = 5.0
                teleop_keyboard.send_goal_joint_space(pathtime)
            elif key_value == '3':  # marker 탐색중
                goal_joint_angle[0] = radians(-30)
                goal_joint_angle[1] = -1.4
                goal_joint_angle[2] = 1.1
                goal_joint_angle[3] = 0.35
                goal_joint_angle[4] = 0.01
                pathtime = 5.0
                teleop_keyboard.send_goal_joint_space(pathtime)
                teleop_keyboard.send_tool_control_request()
            elif key_value == '4':  # marker 탐지시 정면으로 이동
                goal_joint_angle[0] = radians(0)
                goal_joint_angle[1] = 0.0   # 이동위치 정하기
                goal_joint_angle[2] = 0.0
                goal_joint_angle[3] = 0.0
                pathtime = 5.0
                teleop_keyboard.send_goal_joint_space(pathtime)
            elif key_value == '5':  # 이동 후 물건을 집음
                goal_joint_angle[4] = 0.00  # 물체의 크기에 따라 정도 정하기
                teleop_keyboard.send_tool_control_request()
            elif key_value == '6':  # 1번 marker면 특정 위치
                goal_joint_angle[0] = radians(-70)
                goal_joint_angle[1] = 0.0   # 이동위치 정하기
                goal_joint_angle[2] = 0.0
                goal_joint_angle[3] = 0.0
                pathtime = 5.0
                teleop_keyboard.send_goal_joint_space(pathtime)
            elif key_value == '7':  # 2번 marker면 turtle봇 위
                goal_joint_angle[0] = radians(70)
                goal_joint_angle[1] = 0.0   # 이동위치 정하기
                goal_joint_angle[2] = 0.0
                goal_joint_angle[3] = 0.0
                pathtime = 5.0
                teleop_keyboard.send_goal_joint_space(pathtime)
            elif key_value == '8':  # 이동완료 시 놓기
                goal_joint_angle[4] = 0.01
                teleop_keyboard.send_tool_control_request()
            elif key_value == '9':  # 물체를 건드리지 않게 뒤로 빼고 3번으로 돌아가기
                goal_joint_angle[1] = -1.4
                goal_joint_angle[2] = 1.1
                goal_joint_angle[3] = 0.35
                pathtime = 5.0
                teleop_keyboard.send_goal_joint_space(pathtime)   
            else:
                if key_value == '\x03':
                    break
                else:
                    for index in range(0, 7):
                        prev_goal_kinematics_pose[index] = goal_kinematics_pose[index]
                    for index in range(0, 5):
                        prev_goal_joint_angle[index] = goal_joint_angle[index]

    except Exception as e:
        print(e)

    finally:
        if os.name != 'nt':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        teleop_keyboard.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
