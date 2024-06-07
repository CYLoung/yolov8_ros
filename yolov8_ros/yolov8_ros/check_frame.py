import rclpy
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros import TransformListener, TransformException
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration

class TFTestNode(Node):

    def __init__(self):
        super().__init__('tf_test_node')

        # Create a static transform broadcaster
        self.static_broadcaster = StaticTransformBroadcaster(self)

        # Define and broadcast the static transform
        # self.broadcast_static_transform()

        # TF Buffer with increased cache time
        self.tf_buffer = Buffer(cache_time=Duration(seconds=60.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to periodically check the transform
        self.create_timer(2.0, self.check_transform)
    def broadcast_static_transform(self):
        static_transform_stamped = TransformStamped()
        static_transform_stamped.header.stamp = self.get_clock().now().to_msg()
        static_transform_stamped.header.frame_id = 'camera_depth_frame'
        static_transform_stamped.child_frame_id = 'camera_depth_optical_frame'
        static_transform_stamped.transform.translation.x = 0.0
        static_transform_stamped.transform.translation.y = 0.0
        static_transform_stamped.transform.translation.z = 0.0
        static_transform_stamped.transform.rotation.x = -0.5
        static_transform_stamped.transform.rotation.y = 0.4999999999999999
        static_transform_stamped.transform.rotation.z = -0.5
        static_transform_stamped.transform.rotation.w = 0.5000000000000001

        self.static_broadcaster.sendTransform(static_transform_stamped)
        self.get_logger().info(f'Static transform published between camera_depth_frame and camera_depth_optical_frame at time: {static_transform_stamped.header.stamp}')

    def check_transform(self):
        try:
            now = self.get_clock().now().to_msg()
            transform = self.tf_buffer.lookup_transform('camera_depth_frame', 'camera_depth_optical_frame', now, timeout=Duration(seconds=5.0))
            self.get_logger().info(f'Transform available: {transform}')
        except TransformException as ex:
            self.get_logger().error(f'Could not transform: {ex}')

def main():
    rclpy.init()
    node = TFTestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
