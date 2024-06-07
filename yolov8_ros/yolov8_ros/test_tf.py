import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped

class TfListener(Node):

    def __init__(self):
        super().__init__('tf_listener', namespace='my_namespace')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        try:
            # 최신 transform을 요청하기 위해 time에 0을 사용합니다.
            trans = self.tf_buffer.lookup_transform('camera_link', 'camera_depth_optical_frame', rclpy.time.Time(seconds=0))
            self.get_logger().info(f'Got transform: {trans}')
        except Exception as e:
            self.get_logger().warn(f'Could not get transform: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = TfListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
