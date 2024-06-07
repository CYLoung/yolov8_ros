import rclpy
from rclpy.node import Node
from yolov8_msgs.msg import DetectionArray
import tf2_ros
import yaml
from geometry_msgs.msg import TransformStamped

class DetectionTFRecorder(Node):
    def __init__(self):
        super().__init__('detection_tf_recorder')
        self.subscription = self.create_subscription(
            DetectionArray,
            '/detections',
            self.detection_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def detection_callback(self, msg):
        # print(msg)
        for detection in msg.detections:
            # Process each detection to save TF information
            # print(detection)
            self.save_tf(detection)

    def save_tf(self, detection):
        # Save TF information to a file
        tf_data = self.load_tf_file()
        tf_entry = {
            'class_id': detection.class_id,
            'class_name': detection.class_name,
            'score': detection.score,
            'id': detection.id,
            'bbox': {
                'center': {
                    'position': {
                        'x': detection.bbox.center.position.x,
                        'y': detection.bbox.center.position.y
                    },
                    'theta': detection.bbox.center.theta
                },
                'size': {
                    'x': detection.bbox.size.x,
                    'y': detection.bbox.size.y
                }
            }
        }
        print(tf_entry)
        tf_data.append(tf_entry)
        with open('./detections.yaml', 'w') as file:
            yaml.dump(tf_data, file)

    def load_tf_file(self):
        # Load existing TF information from file or initialize if file does not exist
        try:
            with open('./detections.yaml', 'r') as file:
                tf_data = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            tf_data = []
            print(tf_data)
        return tf_data

def main(args=None):
    print("START")
    rclpy.init(args=args)
    detection_tf_recorder = DetectionTFRecorder()
    rclpy.spin(detection_tf_recorder)
    detection_tf_recorder.destroy_node()
    rclpy.shutdown()
if __name__ == "__main__":
    main()