import rclpy
from rclpy.node import Node
from yolov8_msgs.msg import DetectionArray, Detection
import tf2_ros
import yaml
from geometry_msgs.msg import TransformStamped
from math import sqrt
from visualization_msgs.msg import Marker, MarkerArray

class DetectionTFRecorder(Node):
    def __init__(self):
        super().__init__('detection_tf_recorder')
        self.subscription = self.create_subscription(
            DetectionArray,
            '/detections_3d',
            self.detection_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.yaml_file = './detections_3d.yaml'
        # self.publisher = self.create_publisher(DetectionArray, '/detections_3d_published', 10)
        # self.timer = self.create_timer(1.0, self.publish_detections_from_yaml)  # 1초 간격으로 퍼블리시

        self.publisher = self.create_publisher(MarkerArray, '/detection_markers', 10)
        self.timer = self.create_timer(1.0, self.publish_detections_marker_from_yaml)  # 1초 간격으로 퍼블리시

        # update threshold
        self.distance_threshold = 0.5

    def detection_callback(self, msg):
        for detection in msg.detections:
            self.update_or_save_tf(detection)

    def update_or_save_tf(self, detection):
        tf_data = self.load_tf_file()
        existing_entry = self.find_existing_entry(tf_data, detection)

        if existing_entry:
            if 'bbox3d' not in existing_entry:
                existing_entry['bbox3d'] = {'center': {'position': {'x': 0, 'y': 0, 'z': 0}}, 'size': {'x': 0, 'y': 0, 'z': 0}}
            distance = self.calculate_distance(existing_entry['bbox3d']['center']['position'], detection.bbox3d.center.position)
            if distance > self.distance_threshold:
                self.update_entry(existing_entry, detection)
        else:
            self.save_tf(detection, tf_data)

        self.save_tf_data(tf_data)

    def calculate_distance(self, pos1, pos2):
        return sqrt((pos1['x'] - pos2.x) ** 2 + (pos1['y'] - pos2.y) ** 2 + (pos1['z'] - pos2.z) ** 2)

    def find_existing_entry(self, tf_data, detection):
        for entry in tf_data:
            print(entry , detection.class_name)
            if 'class_name' in entry and entry['class_name'] == detection.class_name:
                print("previously detection object!")
                return entry
        return None

    def update_entry(self, existing_entry, detection):
        existing_entry['bbox3d']['center']['position'] = {
            'x': detection.bbox3d.center.position.x,
            'y': detection.bbox3d.center.position.y,
            'z': detection.bbox3d.center.position.z
        }
        existing_entry['bbox3d']['center']['orientation'] = {
            'x': detection.bbox3d.center.orientation.x,
            'y': detection.bbox3d.center.orientation.y,
            'z': detection.bbox3d.center.orientation.z,
            'w': detection.bbox3d.center.orientation.w
        }
        existing_entry['bbox3d']['size'] = {
            'x': detection.bbox3d.size.x,
            'y': detection.bbox3d.size.y,
            'z': detection.bbox3d.size.z
        }

    def save_tf(self, detection, tf_data):
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
            },
            'bbox3d': {
                'center': {
                    'position': {
                        'x': detection.bbox3d.center.position.x,
                        'y': detection.bbox3d.center.position.y,
                        'z': detection.bbox3d.center.position.z
                    },
                    'orientation': {
                        'x': detection.bbox3d.center.orientation.x,
                        'y': detection.bbox3d.center.orientation.y,
                        'z': detection.bbox3d.center.orientation.z,
                        'w': detection.bbox3d.center.orientation.w
                    }
                },
                'size': {
                    'x': detection.bbox3d.size.x,
                    'y': detection.bbox3d.size.y,
                    'z': detection.bbox3d.size.z
                },
                'frame_id': detection.bbox3d.frame_id
            }
        }
        tf_data.append(tf_entry)

    def load_tf_file(self):
        try:
            with open(self.yaml_file, 'r') as file:
                tf_data = yaml.load(file, Loader=yaml.FullLoader)
                if tf_data is None:
                    tf_data = []
        except FileNotFoundError:
            tf_data = []
        return tf_data

    def save_tf_data(self, tf_data):
        with open(self.yaml_file, 'w') as file:
            yaml.dump(tf_data, file)


    def publish_detections_from_yaml(self):
        tf_data = self.load_tf_file()
        msg = DetectionArray()
        for entry in tf_data:
            detection = Detection()
            detection.class_id = entry['class_id']
            detection.class_name = entry['class_name']
            detection.score = entry['score']
            detection.id = entry['id']
            detection.bbox.center.position.x = entry['bbox']['center']['position']['x']
            detection.bbox.center.position.y = entry['bbox']['center']['position']['y']
            detection.bbox.center.theta = entry['bbox']['center']['theta']
            detection.bbox.size.x = entry['bbox']['size']['x']
            detection.bbox.size.y = entry['bbox']['size']['y']
            detection.bbox3d.center.position.x = entry['bbox3d']['center']['position']['x']
            detection.bbox3d.center.position.y = entry['bbox3d']['center']['position']['y']
            detection.bbox3d.center.position.z = entry['bbox3d']['center']['position']['z']
            detection.bbox3d.center.orientation.x = entry['bbox3d']['center']['orientation']['x']
            detection.bbox3d.center.orientation.y = entry['bbox3d']['center']['orientation']['y']
            detection.bbox3d.center.orientation.z = entry['bbox3d']['center']['orientation']['z']
            detection.bbox3d.center.orientation.w = entry['bbox3d']['center']['orientation']['w']
            detection.bbox3d.size.x = entry['bbox3d']['size']['x']
            detection.bbox3d.size.y = entry['bbox3d']['size']['y']
            detection.bbox3d.size.z = entry['bbox3d']['size']['z']
            detection.bbox3d.frame_id = entry['bbox3d']['frame_id']
            msg.detections.append(detection)
        self.publisher.publish(msg)

    def publish_detections_marker_from_yaml(self):
        tf_data = self.load_tf_file()
        marker_array = MarkerArray()
        for i, entry in enumerate(tf_data):
            marker = Marker()
            marker.header.frame_id = entry['bbox3d']['frame_id']
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "detections"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = entry['bbox3d']['center']['position']['x']
            marker.pose.position.y = entry['bbox3d']['center']['position']['y']
            marker.pose.position.z = entry['bbox3d']['center']['position']['z']
            marker.pose.orientation.x = entry['bbox3d']['center']['orientation']['x']
            marker.pose.orientation.y = entry['bbox3d']['center']['orientation']['y']
            marker.pose.orientation.z = entry['bbox3d']['center']['orientation']['z']
            marker.pose.orientation.w = entry['bbox3d']['center']['orientation']['w']
            marker.scale.x = entry['bbox3d']['size']['x']
            marker.scale.y = entry['bbox3d']['size']['y']
            marker.scale.z = entry['bbox3d']['size']['z']
            marker.color.a = 0.8  # 투명도
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            marker_array.markers.append(marker)
        self.publisher.publish(marker_array)
def main(args=None):
    print("START")
    rclpy.init(args=args)
    detection_tf_recorder = DetectionTFRecorder()
    rclpy.spin(detection_tf_recorder)
    detection_tf_recorder.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
