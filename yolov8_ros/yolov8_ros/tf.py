#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pip install transforms3d => python3 tf.py run  
import sys
import time
import math
import numpy as np
import transforms3d

import rclpy
from rclpy.node import Node
import tf2_ros
import geometry_msgs.msg



class ROS2TF():
    def __init__(self,
                 node: Node,
                 verbose=False):

        self._node = node
        self.verbose = verbose

        # TF buffer & listener
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self._node)

        # TF broadcaster (static & dynamic)
        self.broadcaster_static  = tf2_ros.StaticTransformBroadcaster(self._node)
        self.broadcaster_dynamic = tf2_ros.TransformBroadcaster(self._node)


    def getTF(self, reference_frame, target_frame):
        trans = []
        rot   = []
        try:
            # Using the class level buffer
            if self.tf_buffer.can_transform(reference_frame, target_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1)):
                transform = self.tf_buffer.lookup_transform(reference_frame, target_frame, rclpy.time.Time()) # time.Time() recently TF time
                trans = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
                rot = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
                self._node.get_logger().info(f"Transform from {reference_frame} to {target_frame} available.")
            else:
                self._node.get_logger().warn(f"Transform from {reference_frame} to {target_frame} not available.")
        except Exception as e:
            print(e)
        return (np.asarray(trans), np.asarray(rot))


    def publishTF(self, reference_frame, target_frame, translation, rotation=None, static_tf=False):
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = self._node.get_clock().now().to_msg()
        t.header.frame_id = reference_frame
        t.child_frame_id = target_frame
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]

        if rotation is None:
            quaternion = [0.0, 0.0, 0.0, 1.0] # Default identity quaternion
        elif len(rotation) == 3:
            rotation = [math.radians(r) for r in rotation]
            quaternion = transforms3d.euler.euler2quat(*rotation)
        elif len(rotation) == 4:
            quaternion = rotation

        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]

        if static_tf:  self.broadcaster_static.sendTransform(t)
        else:          self.broadcaster_dynamic.sendTransform(t)




    def getDistanceFromTF(self, reference_frame, target_frame):
      trans, rot = self.getTF(reference_frame, target_frame)
      if len(trans) > 0 and len(rot) > 0:
        dist_trans, dist_rot = calcDistance(trans, rot)





    def getScaledTwistFromTF(self, reference_frame, target_frame, scale=1.0, upper_limit_dist=0.1):
      trans, rot = self.getTF(reference_frame, target_frame)

      if len(trans) > 0 and len(rot) > 0:
        calcScaledTwist(trans, rot, scale, upper_limit_dist)



def calcScaledTwist(trans_in_meter, rot_in_quat, scale=1.0, upper_limit_dist=0.1):
  scaled_trans = trans_in_meter * scale
  scaled_rot = rot_in_quat # TODO: rotation에 대한 scale 적용?

  trans_dist = np.linalg.norm(scaled_trans)
  scaled_trans *= scale
  if trans_dist > upper_limit_dist:
      scale = upper_limit_dist / trans_dist
      scaled_trans *= scale
  # print(f'Scaled twist(scale={scale}): {trans_in_meter} -> {scaled_trans}')
  return (scaled_trans, scaled_rot)


def calcDistance(trans_in_meter, rot_in_quat):
  def quaternion_distance(q1, q2):
    dot_product = np.dot(q1, q2)  # 쿼터니언 내적 계산
    return np.arccos(2 * dot_product**2 - 1) # 내적 결과를 사용하여 각도 차이 계산

  translation_distance = np.linalg.norm(trans_in_meter)
  rotation_distance = quaternion_distance(rot_in_quat, [0, 0, 0, 1])
  # print(f'Distance in translation: {translation_distance}, \nDistance in rotation: {rotation_distance}')
  return (translation_distance, rotation_distance)



if __name__ == "__main__":
    rclpy.init()
    node = Node("ros2_camera_topic")

    tf = ROS2TF(node, verbose=True)


    def test_tf_functions():
        reference_frame = 'camera_depth_optical_frame' # Robot base
        target_frame = 'camera_depth_frame' # Camera topic
        # translation = [1.037, -0.146, -0.041]
        # tf.publishTF(reference_frame, target_frame, translation)
        transform = tf.getTF(reference_frame, target_frame)
        if transform:
            print("Transform obtained:", transform)

        tf.getDistanceFromTF(reference_frame, target_frame)
        tf.getScaledTwistFromTF(reference_frame, target_frame, scale=0.9)

    timer = node.create_timer(1, test_tf_functions)

    rclpy.spin(node)  # Using spin instead of MultiThreadedExecutor for simplicity



    # Spin the node in background thread(s)
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor.spin()

    rclpy.shutdown()
    exit(0)
