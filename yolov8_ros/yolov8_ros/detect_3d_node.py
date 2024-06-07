# # # Copyright (C) 2023  Miguel Ángel González Santamarta

# # # This program is free software: you can redistribute it and/or modify
# # # it under the terms of the GNU General Public License as published by
# # # the Free Software Foundation, either version 3 of the License, or
# # # (at your option) any later version.

# # # This program is distributed in the hope that it will be useful,
# # # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# # # GNU General Public License for more details.

# # # You should have received a copy of the GNU General Public License
# # # along with this program.  If not, see <https://www.gnu.org/licenses/>.

# import asyncio
# import numpy as np
# from typing import List, Tuple

# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile
# from rclpy.qos import QoSHistoryPolicy
# from rclpy.qos import QoSDurabilityPolicy
# from rclpy.qos import QoSReliabilityPolicy
# from rclpy.duration import Duration

# import message_filters
# from cv_bridge import CvBridge
# from tf2_ros.buffer import Buffer
# from tf2_ros import TransformException
# from tf2_ros.transform_listener import TransformListener
# from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

# from sensor_msgs.msg import CameraInfo, Image
# from geometry_msgs.msg import TransformStamped
# from yolov8_msgs.msg import Detection
# from yolov8_msgs.msg import DetectionArray
# from yolov8_msgs.msg import KeyPoint3D
# from yolov8_msgs.msg import KeyPoint3DArray
# from yolov8_msgs.msg import BoundingBox3D

# from .tf import ROS2TF


# class Detect3DNode(Node):

#     def __init__(self) -> None:
#         super().__init__("bbox3d_node")
#         # #@LCY
#         # # Create a static transform broadcaster
#         # self.static_broadcaster = StaticTransformBroadcaster(self)

#         # # Define the static transform
#         # static_transform_stamped = TransformStamped()

#         # static_transform_stamped.header.stamp = self.get_clock().now().to_msg()
#         # static_transform_stamped.header.frame_id = 'camera_depth_optical_frame'
#         # static_transform_stamped.child_frame_id = 'camera_depth_frame'

#         # static_transform_stamped.transform.translation.x = 0.0
#         # static_transform_stamped.transform.translation.y = 0.0
#         # static_transform_stamped.transform.translation.z = 0.0
#         # static_transform_stamped.transform.rotation.x = 0.0
#         # static_transform_stamped.transform.rotation.y = 0.0
#         # static_transform_stamped.transform.rotation.z = 0.0
#         # static_transform_stamped.transform.rotation.w = 1.0

#         # # Broadcast the static transform
#         # self.static_broadcaster.sendTransform(static_transform_stamped)
#         # self.get_logger().info('Static transform published between camera_depth_frame and camera_depth_optical_frame')
#         # #@LCYE

#         # self.tf = ROS2TF(self, verbose=True)

#         # parameters
#         self.declare_parameter("target_frame", "base_link")
#         self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value

#         # Log the target frame to verify it's correctly set
#         self.get_logger().info(f"Target frame set to: {self.target_frame}")
        
#         self.declare_parameter("maximum_detection_threshold", 0.3)
#         self.maximum_detection_threshold = self.get_parameter(
#             "maximum_detection_threshold").get_parameter_value().double_value

#         self.declare_parameter("depth_image_units_divisor", 1000)
#         self.depth_image_units_divisor = self.get_parameter(
#             "depth_image_units_divisor").get_parameter_value().integer_value

#         self.declare_parameter("depth_image_reliability",
#                                QoSReliabilityPolicy.BEST_EFFORT)
#         depth_image_qos_profile = QoSProfile(
#             reliability=self.get_parameter(
#                 "depth_image_reliability").get_parameter_value().integer_value,
#             history=QoSHistoryPolicy.KEEP_LAST,
#             durability=QoSDurabilityPolicy.VOLATILE,
#             depth=1
#         )

#         self.declare_parameter("depth_info_reliability",
#                                QoSReliabilityPolicy.BEST_EFFORT)
#         depth_info_qos_profile = QoSProfile(
#             reliability=self.get_parameter(
#                 "depth_info_reliability").get_parameter_value().integer_value,
#             history=QoSHistoryPolicy.KEEP_LAST,
#             durability=QoSDurabilityPolicy.VOLATILE,
#             depth=1
#         )

#         # # aux
#         # self.tf_buffer = Buffer()
#         # TF Buffer with increased cache time
#         # cache_time = Duration(seconds=60.0)  # 캐시 지속 시간을 10초로 설정
#         # self.tf_buffer = Buffer(cache_time=cache_time)

#         self.tf_buffer = Buffer()
#         self.tf_listener = TransformListener(self.tf_buffer, self)
#         self.cv_bridge = CvBridge()

#         # pubs
#         self._pub = self.create_publisher(DetectionArray, "detections_3d", 10)

#         # subs
#         self.depth_sub = message_filters.Subscriber(
#             self, Image, "depth_image",
#             qos_profile=depth_image_qos_profile)
#         self.depth_info_sub = message_filters.Subscriber(
#             self, CameraInfo, "depth_info",
#             qos_profile=depth_info_qos_profile)
#         self.detections_sub = message_filters.Subscriber(
#             self, DetectionArray, "detections")

#         self._synchronizer = message_filters.ApproximateTimeSynchronizer(
#             (self.depth_sub, self.depth_info_sub, self.detections_sub), 10, 0.5)
#         self._synchronizer.registerCallback(self.on_detections)

#     def on_detections(
#         self,
#         depth_msg: Image,
#         depth_info_msg: CameraInfo,
#         detections_msg: DetectionArray,
#     ) -> None:

#         new_detections_msg = DetectionArray()
#         new_detections_msg.header = detections_msg.header
#         new_detections_msg.detections = self.process_detections(
#             depth_msg, depth_info_msg, detections_msg)
#         self.get_logger().info(f"#### on_detections: {detections_msg}")
#         self._pub.publish(new_detections_msg)

#     def process_detections(
#         self,
#         depth_msg: Image,
#         depth_info_msg: CameraInfo,
#         detections_msg: DetectionArray
#     ) -> List[Detection]:

#         # check if there are detections
#         if not detections_msg.detections:
#             return []
#         # @check : done
#         # self.get_logger().info(f"frame_id initial get: {depth_info_msg.header.frame_id}")
#         # Format: Z16, Width: 848, Height: 480, FPS: 30
#         # transform = self.get_transform(depth_info_msg.header.frame_id)
#         timestamp = self.get_clock().now().to_msg()
#         reference_frame = 'camera_depth_optical_frame'
#         transform = self.get_transform(reference_frame, timestamp)
#         if transform is None:
#             return []

#         new_detections = []
#         depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg)

#         for detection in detections_msg.detections:
#             bbox3d = self.convert_bb_to_3d(
#                 depth_image, depth_info_msg, detection)

#             if bbox3d is not None:
#                 new_detections.append(detection)

#                 bbox3d = Detect3DNode.transform_3d_box(
#                     bbox3d, transform[0], transform[1])
#                 bbox3d.frame_id = self.target_frame
#                 new_detections[-1].bbox3d = bbox3d

#                 if detection.keypoints.data:
#                     keypoints3d = self.convert_keypoints_to_3d(
#                         depth_image, depth_info_msg, detection)
#                     keypoints3d = Detect3DNode.transform_3d_keypoints(
#                         keypoints3d, transform[0], transform[1])
#                     keypoints3d.frame_id = self.target_frame
#                     new_detections[-1].keypoints3d = keypoints3d

#         return new_detections

#     def convert_bb_to_3d(
#         self,
#         depth_image: np.ndarray,
#         depth_info: CameraInfo,
#         detection: Detection
#     ) -> BoundingBox3D:

#         # crop depth image by the 2d BB
#         center_x = int(detection.bbox.center.position.x)
#         center_y = int(detection.bbox.center.position.y)
#         size_x = int(detection.bbox.size.x)
#         size_y = int(detection.bbox.size.y)

#         u_min = max(center_x - size_x // 2, 0)
#         u_max = min(center_x + size_x // 2, depth_image.shape[1] - 1)
#         v_min = max(center_y - size_y // 2, 0)
#         v_max = min(center_y + size_y // 2, depth_image.shape[0] - 1)

#         roi = depth_image[v_min:v_max, u_min:u_max] / \
#             self.depth_image_units_divisor  # convert to meters
#         if not np.any(roi):
#             return None

#         # find the z coordinate on the 3D BB
#         bb_center_z_coord = depth_image[int(center_y)][int(
#             center_x)] / self.depth_image_units_divisor
#         z_diff = np.abs(roi - bb_center_z_coord)
#         mask_z = z_diff <= self.maximum_detection_threshold
#         if not np.any(mask_z):
#             return None

#         roi_threshold = roi[mask_z]
#         z_min, z_max = np.min(roi_threshold), np.max(roi_threshold)
#         z = (z_max + z_min) / 2
#         if z == 0:
#             return None

#         # project from image to world space
#         k = depth_info.k
#         px, py, fx, fy = k[2], k[5], k[0], k[4]
#         x = z * (center_x - px) / fx
#         y = z * (center_y - py) / fy
#         w = z * (size_x / fx)
#         h = z * (size_y / fy)

#         # create 3D BB
#         msg = BoundingBox3D()
#         msg.center.position.x = x
#         msg.center.position.y = y
#         msg.center.position.z = z
#         msg.size.x = w
#         msg.size.y = h
#         msg.size.z = float(z_max - z_min)

#         return msg

#     def convert_keypoints_to_3d(
#         self,
#         depth_image: np.ndarray,
#         depth_info: CameraInfo,
#         detection: Detection
#     ) -> KeyPoint3DArray:

#         # build an array of 2d keypoints
#         keypoints_2d = np.array([[p.point.x, p.point.y]
#                                 for p in detection.keypoints.data], dtype=np.int16)
#         u = np.array(keypoints_2d[:, 1]).clip(0, depth_info.height - 1)
#         v = np.array(keypoints_2d[:, 0]).clip(0, depth_info.width - 1)

#         # sample depth image and project to 3D
#         z = depth_image[u, v]
#         k = depth_info.k
#         px, py, fx, fy = k[2], k[5], k[0], k[4]
#         x = z * (v - px) / fx
#         y = z * (u - py) / fy
#         points_3d = np.dstack([x, y, z]).reshape(-1, 3) / \
#             self.depth_image_units_divisor  # convert to meters

#         # generate message
#         msg_array = KeyPoint3DArray()
#         for p, d in zip(points_3d, detection.keypoints.data):
#             if not np.isnan(p).any():
#                 msg = KeyPoint3D()
#                 msg.point.x = p[0]
#                 msg.point.y = p[1]
#                 msg.point.z = p[2]
#                 msg.id = d.id
#                 msg.score = d.score
#                 msg_array.data.append(msg)

#         return msg_array

#     def get_transform(self, frame_id: str, timestamp: rclpy.time.Time) -> Tuple[np.ndarray]:
#         rotation = None
#         translation = None
#         self.get_logger().info(f"frame_id: {frame_id}")
#         self.get_logger().info(f"target_frame: {self.target_frame}")
#         # tf_future = self.tf_buffer.wait_for_transform_async(
#         #     target_frame=self.target_frame,
#         #     source_frame=frame_id,
#         #     time=rclpy.time.Time()
#         # )
#         # rclpy.spin_until_future_complete(self, tf_future)

#         try:
#             self.get_logger().info('1------------------------------------------')
#             self.tf.getTF('/camera_depth_optical_frame', '/camera_depth_frame')
#             self.get_logger().info('2------------------------------------------')
#             can_transform = self.tf_buffer.can_transform(
#                 self.target_frame, frame_id, timestamp, timeout=rclpy.duration.Duration(seconds=5))
#             if not can_transform:
#                 self.get_logger().error(f"Cannot transform from {frame_id} to {self.target_frame} at time {timestamp}")
#                 return None
            
#             self.get_logger().info(f"Transform available from {frame_id} to {self.target_frame} at time {timestamp}")

#             transform: TransformStamped = self.tf_buffer.lookup_transform(
#                 self.target_frame,
#                 frame_id,
#                 timestamp,  # 메시지의 타임스탬프를 사용
#                 timeout=rclpy.duration.Duration(seconds=5))  # 타임아웃 시간 증가

#             translation = np.array([transform.transform.translation.x,
#                                     transform.transform.translation.y,
#                                     transform.transform.translation.z])

#             rotation = np.array([transform.transform.rotation.w,
#                                  transform.transform.rotation.x,
#                                  transform.transform.rotation.y,
#                                  transform.transform.rotation.z])

#             return translation, rotation

#         except TransformException as ex:
#             self.get_logger().error(f"Could not transform from {frame_id} to {self.target_frame}: {ex}")
#             return None

#     @staticmethod
#     def transform_3d_box(
#         bbox: BoundingBox3D,
#         translation: np.ndarray,
#         rotation: np.ndarray
#     ) -> BoundingBox3D:

#         # position
#         position = Detect3DNode.qv_mult(
#             rotation,
#             np.array([bbox.center.position.x,
#                       bbox.center.position.y,
#                       bbox.center.position.z])
#         ) + translation

#         bbox.center.position.x = position[0]
#         bbox.center.position.y = position[1]
#         bbox.center.position.z = position[2]

#         # size
#         size = Detect3DNode.qv_mult(
#             rotation,
#             np.array([bbox.size.x,
#                       bbox.size.y,
#                       bbox.size.z])
#         )

#         bbox.size.x = abs(size[0])
#         bbox.size.y = abs(size[1])
#         bbox.size.z = abs(size[2])

#         return bbox

#     @staticmethod
#     def transform_3d_keypoints(
#         keypoints: KeyPoint3DArray,
#         translation: np.ndarray,
#         rotation: np.ndarray,
#     ) -> KeyPoint3DArray:

#         for point in keypoints.data:
#             position = Detect3DNode.qv_mult(
#                 rotation,
#                 np.array([
#                     point.point.x,
#                     point.point.y,
#                     point.point.z
#                 ])
#             ) + translation

#             point.point.x = position[0]
#             point.point.y = position[1]
#             point.point.z = position[2]

#         return keypoints

#     @staticmethod
#     def qv_mult(q: np.ndarray, v: np.ndarray) -> np.ndarray:
#         q = np.array(q, dtype=np.float64)
#         v = np.array(v, dtype=np.float64)
#         qvec = q[1:]
#         uv = np.cross(qvec, v)
#         uuv = np.cross(qvec, uv)
#         return v + 2 * (uv * q[0] + uuv)


# # class ROS2TF():
# #     def __init__(self,
# #                  node: Node,
# #                  verbose=False):

# #         self._node = node
# #         self.verbose = verbose

# #         # TF buffer & listener
# #         self.tf_buffer   = Buffer()
# #         self.tf_listener = TransformListener(self.tf_buffer, self._node)

# #         # TF broadcaster (static & dynamic)
# #         self.broadcaster_static  = StaticTransformBroadcaster(self._node)


# #     def getTF(self, reference_frame, target_frame):
# #         trans = []
# #         rot   = []
# #         try:
# #             # Using the class level buffer
# #             if self.tf_buffer.can_transform(reference_frame, target_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1)):
# #                 transform = self.tf_buffer.lookup_transform(reference_frame, target_frame, rclpy.time.Time()) # time.Time() recently TF time
# #                 trans = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
# #                 rot = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
# #             else:
# #                 self._node.get_logger().info(f"<test>Transform from {reference_frame} to {target_frame} not available.")
# #         except Exception as e:
# #             self._node.get_logger().error(f"<test>{e}")
# #         return (np.asarray(trans), np.asarray(rot))




# def main_test():
#     rclpy.init()
#     # node = Detect3DNode()
#     node = Node("ros2_camera_topic")
    
#     tf = ROS2TF(node, verbose=True)
#     node.get_logger().info(f"#### {dir(node)}")
#     def test_tf_functions():
#         reference_frame = 'camera_depth_optical_frame' # Robot base
#         target_frame = 'camera_depth_frame' # Camera topic
#         # translation = [1.037, -0.146, -0.041]
#         # tf.publishTF(reference_frame, target_frame, translation)
#         transform = tf.getTF(reference_frame, target_frame)
#     timer = node.create_timer(1, test_tf_functions)

#     rclpy.spin(node)
#     # Spin the node in background thread(s)
#     executor = rclpy.executors.MultiThreadedExecutor(2)
#     executor.add_node(node)
#     executor.spin()

#     rclpy.shutdown()
#     exit(0)

# def main():
#     rclpy.init()
#     node = Detect3DNode()

#     rclpy.spin(node)
#     # Spin the node in background thread(s)
#     executor = rclpy.executors.MultiThreadedExecutor(2)
#     executor.add_node(node)
#     executor.spin()

#     rclpy.shutdown()
#     exit(0)

# if __name__ == "__main__":
#     main()

# Copyright (C) 2023  Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

import message_filters
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
from yolov8_msgs.msg import Detection
from yolov8_msgs.msg import DetectionArray
from yolov8_msgs.msg import KeyPoint3D
from yolov8_msgs.msg import KeyPoint3DArray
from yolov8_msgs.msg import BoundingBox3D


class Detect3DNode(Node):

    def __init__(self) -> None:
        super().__init__("bbox3d_node")

        # parameters
        self.declare_parameter("target_frame", "base_link")
        self.target_frame = self.get_parameter(
            "target_frame").get_parameter_value().string_value

        self.declare_parameter("maximum_detection_threshold", 0.3)
        self.maximum_detection_threshold = self.get_parameter(
            "maximum_detection_threshold").get_parameter_value().double_value

        self.declare_parameter("depth_image_units_divisor", 1000)
        self.depth_image_units_divisor = self.get_parameter(
            "depth_image_units_divisor").get_parameter_value().integer_value

        self.declare_parameter("depth_image_reliability",
                               QoSReliabilityPolicy.BEST_EFFORT)
        depth_image_qos_profile = QoSProfile(
            reliability=self.get_parameter(
                "depth_image_reliability").get_parameter_value().integer_value,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.declare_parameter("depth_info_reliability",
                               QoSReliabilityPolicy.BEST_EFFORT)
        depth_info_qos_profile = QoSProfile(
            reliability=self.get_parameter(
                "depth_info_reliability").get_parameter_value().integer_value,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # aux
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cv_bridge = CvBridge()

        # pubs
        self._pub = self.create_publisher(DetectionArray, "detections_3d", 10)

        # subs
        self.depth_sub = message_filters.Subscriber(
            self, Image, "depth_image",
            qos_profile=depth_image_qos_profile)
        self.depth_info_sub = message_filters.Subscriber(
            self, CameraInfo, "depth_info",
            qos_profile=depth_info_qos_profile)
        self.detections_sub = message_filters.Subscriber(
            self, DetectionArray, "detections")

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.depth_sub, self.depth_info_sub, self.detections_sub), 10, 0.5)
        self._synchronizer.registerCallback(self.on_detections)

    def on_detections(
        self,
        depth_msg: Image,
        depth_info_msg: CameraInfo,
        detections_msg: DetectionArray,
    ) -> None:

        new_detections_msg = DetectionArray()
        new_detections_msg.header = detections_msg.header
        new_detections_msg.detections = self.process_detections(
            depth_msg, depth_info_msg, detections_msg)
        self._pub.publish(new_detections_msg)

    def process_detections(
        self,
        depth_msg: Image,
        depth_info_msg: CameraInfo,
        detections_msg: DetectionArray
    ) -> List[Detection]:

        # check if there are detections
        if not detections_msg.detections:
            return []

        transform = self.get_transform(depth_info_msg.header.frame_id)

        if transform is None:
            return []

        new_detections = []
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg)

        for detection in detections_msg.detections:
            bbox3d = self.convert_bb_to_3d(
                depth_image, depth_info_msg, detection)

            if bbox3d is not None:
                new_detections.append(detection)

                bbox3d = Detect3DNode.transform_3d_box(
                    bbox3d, transform[0], transform[1])
                bbox3d.frame_id = self.target_frame
                new_detections[-1].bbox3d = bbox3d

                if detection.keypoints.data:
                    keypoints3d = self.convert_keypoints_to_3d(
                        depth_image, depth_info_msg, detection)
                    keypoints3d = Detect3DNode.transform_3d_keypoints(
                        keypoints3d, transform[0], transform[1])
                    keypoints3d.frame_id = self.target_frame
                    new_detections[-1].keypoints3d = keypoints3d

        return new_detections

    def convert_bb_to_3d(
        self,
        depth_image: np.ndarray,
        depth_info: CameraInfo,
        detection: Detection
    ) -> BoundingBox3D:

        # crop depth image by the 2d BB
        center_x = int(detection.bbox.center.position.x)
        center_y = int(detection.bbox.center.position.y)
        size_x = int(detection.bbox.size.x)
        size_y = int(detection.bbox.size.y)

        u_min = max(center_x - size_x // 2, 0)
        u_max = min(center_x + size_x // 2, depth_image.shape[1] - 1)
        v_min = max(center_y - size_y // 2, 0)
        v_max = min(center_y + size_y // 2, depth_image.shape[0] - 1)

        roi = depth_image[v_min:v_max, u_min:u_max] / \
            self.depth_image_units_divisor  # convert to meters
        if not np.any(roi):
            return None

        # find the z coordinate on the 3D BB
        bb_center_z_coord = depth_image[int(center_y)][int(
            center_x)] / self.depth_image_units_divisor
        z_diff = np.abs(roi - bb_center_z_coord)
        mask_z = z_diff <= self.maximum_detection_threshold
        if not np.any(mask_z):
            return None

        roi_threshold = roi[mask_z]
        z_min, z_max = np.min(roi_threshold), np.max(roi_threshold)
        z = (z_max + z_min) / 2
        if z == 0:
            return None

        # project from image to world space
        k = depth_info.k
        px, py, fx, fy = k[2], k[5], k[0], k[4]
        x = z * (center_x - px) / fx
        y = z * (center_y - py) / fy
        w = z * (size_x / fx)
        h = z * (size_y / fy)

        # create 3D BB
        msg = BoundingBox3D()
        msg.center.position.x = x
        msg.center.position.y = y
        msg.center.position.z = z
        msg.size.x = w
        msg.size.y = h
        msg.size.z = float(z_max - z_min)

        return msg

    def convert_keypoints_to_3d(
        self,
        depth_image: np.ndarray,
        depth_info: CameraInfo,
        detection: Detection
    ) -> KeyPoint3DArray:

        # build an array of 2d keypoints
        keypoints_2d = np.array([[p.point.x, p.point.y]
                                for p in detection.keypoints.data], dtype=np.int16)
        u = np.array(keypoints_2d[:, 1]).clip(0, depth_info.height - 1)
        v = np.array(keypoints_2d[:, 0]).clip(0, depth_info.width - 1)

        # sample depth image and project to 3D
        z = depth_image[u, v]
        k = depth_info.k
        px, py, fx, fy = k[2], k[5], k[0], k[4]
        x = z * (v - px) / fx
        y = z * (u - py) / fy
        points_3d = np.dstack([x, y, z]).reshape(-1, 3) / \
            self.depth_image_units_divisor  # convert to meters

        # generate message
        msg_array = KeyPoint3DArray()
        for p, d in zip(points_3d, detection.keypoints.data):
            if not np.isnan(p).any():
                msg = KeyPoint3D()
                msg.point.x = p[0]
                msg.point.y = p[1]
                msg.point.z = p[2]
                msg.id = d.id
                msg.score = d.score
                msg_array.data.append(msg)

        return msg_array

    def get_transform(self, frame_id: str) -> Tuple[np.ndarray]:
        # transform position from image frame to target_frame
        rotation = None
        translation = None

        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.target_frame,
                frame_id,
                rclpy.time.Time())

            translation = np.array([transform.transform.translation.x,
                                    transform.transform.translation.y,
                                    transform.transform.translation.z])

            rotation = np.array([transform.transform.rotation.w,
                                 transform.transform.rotation.x,
                                 transform.transform.rotation.y,
                                 transform.transform.rotation.z])

            return translation, rotation

        except TransformException as ex:
            self.get_logger().error(f"Could not transform: {ex}")
            return None

    @staticmethod
    def transform_3d_box(
        bbox: BoundingBox3D,
        translation: np.ndarray,
        rotation: np.ndarray
    ) -> BoundingBox3D:

        # position
        position = Detect3DNode.qv_mult(
            rotation,
            np.array([bbox.center.position.x,
                      bbox.center.position.y,
                      bbox.center.position.z])
        ) + translation

        bbox.center.position.x = position[0]
        bbox.center.position.y = position[1]
        bbox.center.position.z = position[2]

        # size
        size = Detect3DNode.qv_mult(
            rotation,
            np.array([bbox.size.x,
                      bbox.size.y,
                      bbox.size.z])
        )

        bbox.size.x = abs(size[0])
        bbox.size.y = abs(size[1])
        bbox.size.z = abs(size[2])

        return bbox

    @staticmethod
    def transform_3d_keypoints(
        keypoints: KeyPoint3DArray,
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> KeyPoint3DArray:

        for point in keypoints.data:
            position = Detect3DNode.qv_mult(
                rotation,
                np.array([
                    point.point.x,
                    point.point.y,
                    point.point.z
                ])
            ) + translation

            point.point.x = position[0]
            point.point.y = position[1]
            point.point.z = position[2]

        return keypoints

    @staticmethod
    def qv_mult(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q = np.array(q, dtype=np.float64)
        v = np.array(v, dtype=np.float64)
        qvec = q[1:]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        return v + 2 * (uv * q[0] + uuv)


def main():
    rclpy.init()
    node = Detect3DNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()