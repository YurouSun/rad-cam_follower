# mmwave_6843_driver.py
import os
import rclpy
from rclpy.node import Node
import serial
import time
import sys
import struct
from std_msgs.msg import Header
from ament_index_python.packages import get_package_share_directory
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from ti_mmwave_rospkg_msgs.msg import RadarTrackContents, RadarTrackArray
from visualization_msgs.msg import Marker, MarkerArray


class MobileTrackerDriver(Node):
    def __init__(self):
        # The node name can be anything you like
        super().__init__('mmwave_6843_driver')

        # --- ROS2 Parameter Setup ---
        # The package name must match your folder name, e.g., 'mobile_tracker'
        pkg_share_path = get_package_share_directory('mobile_tracker')
        default_config_path = os.path.join(pkg_share_path, 'cfg', 'Mobile_Tracker_car.cfg')

        self.declare_parameter('config_file', default_config_path)
        self.declare_parameter('cli_port', '/dev/ttyUSB0') # Default for Linux
        self.declare_parameter('data_port', '/dev/ttyUSB1')

        self.cli_port_name = self.get_parameter('cli_port').get_parameter_value().string_value
        self.data_port_name = self.get_parameter('data_port').get_parameter_value().string_value
        self.config_file_path = self.get_parameter('config_file').get_parameter_value().string_value
        self.get_logger().info(f"Using config file: {self.config_file_path}")

        # --- Publisher Setup ---
        self.tracker_pub_ = self.create_publisher(RadarTrackArray, 'ti_mmwave/radar_track_array', 10)
        self.point_cloud_pub_ = self.create_publisher(PointCloud2, '/ti_mmwave/radar_scan_pcl', 10)
        self.marker_pub_ = self.create_publisher(MarkerArray, '/ti_mmwave/radar_track_marker_array', 10)

        self.cli_port = None
        self.data_port = None

    def setup_radar(self):
        """Configures and starts the radar, returns True on success."""
        try:
            self.get_logger().info(f"Opening CLI port: {self.cli_port_name}...")
            self.cli_port = serial.Serial(self.cli_port_name, 115200, timeout=1)
            self.cli_port.reset_input_buffer()
            time.sleep(1)

            self.get_logger().info(f"Reading config file: {self.config_file_path}...")
            with open(self.config_file_path, 'r') as f:
                for command in f:
                    command = command.strip()
                    if command and not command.startswith('%'):
                        self.cli_port.write((command + '\n').encode())
                        self.cli_port.read_until(b'mmwDemo:/>')
            self.get_logger().info("Config sent.")

            self.get_logger().info(f"Opening DATA port: {self.data_port_name}...")
            self.data_port = serial.Serial(self.data_port_name, 921600, timeout=0.01)
            self.get_logger().info("Radar setup successful!")
            return True
        except (FileNotFoundError, serial.SerialException) as e:
            self.get_logger().error(f"Radar setup failed: {e}")
            return False

    def run(self):
        """Main loop to receive and process data."""
        if not self.setup_radar():
            return

        byte_buffer = b''
        magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'
        
        last_debug_time = time.time()
        bytes_received_total = 0

        while rclpy.ok():
            try:
                data = self.data_port.read(2048)
                if len(data) > 0:
                    byte_buffer += data
                    bytes_received_total += len(data)
                
                # Debug log every 5 seconds
                if time.time() - last_debug_time > 5.0:
                    self.get_logger().info(f"Data stream status: Buffer size={len(byte_buffer)}, Total bytes received={bytes_received_total}")
                    last_debug_time = time.time()

                magic_word_index = byte_buffer.find(magic_word)
                if magic_word_index != -1:
                    byte_buffer = byte_buffer[magic_word_index:]
                    header_length = 40
                    if len(byte_buffer) >= header_length:
                        header_format = '<8xIIIIIIII'
                        header_tuple = struct.unpack(header_format, byte_buffer[:header_length])
                        total_packet_len = header_tuple[1]
                        if len(byte_buffer) >= total_packet_len:
                            self.parse_frame(byte_buffer[:total_packet_len])
                            byte_buffer = byte_buffer[total_packet_len:]
                elif len(byte_buffer) > 4096: # Prevent buffer from growing indefinitely if no magic word
                     byte_buffer = byte_buffer[-2048:]
                     
            except serial.SerialException as e:
                self.get_logger().error(f"Serial error: {e}. Attempting to reconnect...")
                try:
                    if self.data_port and self.data_port.is_open:
                        self.data_port.close()
                    if self.cli_port and self.cli_port.is_open:
                        self.cli_port.close()
                except Exception as close_e:
                    self.get_logger().warn(f"Error closing ports: {close_e}")
                
                time.sleep(2)
                if self.setup_radar():
                    self.get_logger().info("Reconnected successfully.")
                    byte_buffer = b'' # Clear buffer
                else:
                    self.get_logger().error("Reconnect failed, retrying in 2s...")
                    time.sleep(2)

            except Exception as e:
                self.get_logger().warn(f"Error in main loop: {e}")
                time.sleep(1)

    def parse_frame(self, frame_bytes):
        """Parses a complete data frame and calls the appropriate TLV parser."""
        header_length = 40
        num_tlvs = struct.unpack('<32xI4x', frame_bytes[:header_length])[0]
        current_pos = header_length

        # Dictionaries to hold data for the current frame
        frame_data = {
            'point_cloud': None,
            'point_type': None, # 'spherical' or 'cartesian'
            'side_info': None,
            'target_indices': None,
            'tracks': [] # Store tracks here
        }

        for _ in range(num_tlvs):
            if current_pos + 8 > len(frame_bytes):
                break
            tlv_type, tlv_len = struct.unpack('<II', frame_bytes[current_pos:current_pos + 8])
            tlv_value_bytes = frame_bytes[current_pos + 8 : current_pos + 8 + tlv_len]
            
            # Debug: Log TLV types occasionally
            self.get_logger().info(f"Found TLV type: {tlv_type}, len: {tlv_len}") 
            
            if tlv_type == 1000: # Point Cloud (Spherical)
                frame_data['point_cloud'] = self.parse_point_cloud(tlv_value_bytes)
                frame_data['point_type'] = 'spherical'
            elif tlv_type == 1: # Detected Points (Cartesian)
                frame_data['point_cloud'] = self.parse_point_cloud_type1(tlv_value_bytes)
                frame_data['point_type'] = 'cartesian'
            elif tlv_type == 7: # Side Info
                frame_data['side_info'] = self.parse_side_info(tlv_value_bytes)
            elif tlv_type == 1011: # Target Index
                frame_data['target_indices'] = self.parse_target_index(tlv_value_bytes)
            elif tlv_type == 1010:  # Tracked target list
                frame_data['tracks'] = self.parse_targets(tlv_value_bytes)
            elif tlv_type == 1020: # Side Info for Tracker? Or something else.
                 pass

            current_pos += 8 + tlv_len
        
        # Debug: Log if we got any points or tracks
        if frame_data['point_cloud']:
             self.get_logger().info(f"Frame has {len(frame_data['point_cloud'])} points")
        if frame_data['tracks']:
             self.get_logger().info(f"Frame has {len(frame_data['tracks'])} tracks")
        elif num_tlvs > 0 and not frame_data['point_cloud']:
             self.get_logger().info(f"Frame has {num_tlvs} TLVs but no points/tracks parsed.")

        # Publish Point Cloud
        if frame_data['point_cloud'] is not None:
            self.publish_point_cloud(frame_data)
            
        # Publish Tracks (Always publish, even if empty)
        self.publish_tracks(frame_data['tracks'])

    def parse_targets(self, tlv_value_bytes):
        """Parses the target list TLV and returns a list of RadarTrackContents."""
        target_struct_format = '<I27f'
        target_size = 112
        num_targets = len(tlv_value_bytes) // target_size
        
        tracks_list = []
        for i in range(num_targets):
            try:
                target_data = struct.unpack(target_struct_format, tlv_value_bytes[i*target_size : (i+1)*target_size])
                obj = RadarTrackContents()
                obj.tid = target_data[0]
                obj.posx = -1 * target_data[1]
                obj.posy = target_data[2]
                obj.posz = -1 * target_data[3]
                obj.velx = -1 * target_data[4]
                obj.vely = target_data[5]
                obj.velz = -1 * target_data[6]
                obj.accx = -1 * target_data[7]
                obj.accy = target_data[8]
                obj.accz = -1 * target_data[9]
                tracks_list.append(obj)
            except struct.error:
                continue
        return tracks_list

    def publish_tracks(self, tracks_list):
        """Publishes the RadarTrackArray message."""
        msg = RadarTrackArray()
        msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="ti_mmwave_0")
        msg.num_tracks = len(tracks_list)
        msg.track = tracks_list
        
        self.tracker_pub_.publish(msg)
        
        if tracks_list:
            self.get_logger().info(f"Publishing {len(tracks_list)} tracked targets...")
            self.publish_markers(msg)
        else:
            pass

    def parse_point_cloud_type1(self, tlv_value_bytes):
        """Parse Cartesian point cloud (TLV Type 1)"""
        point_struct_format = '<ffff'  # 4 floats: x, y, z, doppler
        point_size = struct.calcsize(point_struct_format)
        num_points = len(tlv_value_bytes) // point_size
        points = []
        for i in range(num_points):
            point_data = struct.unpack(point_struct_format, tlv_value_bytes[i*point_size : (i+1)*point_size])
            points.append(point_data)
        
        # Debug: Print first few points to check coordinates
        if points:
            x, y, z, d = points[0]
            self.get_logger().info(f"DEBUG Point[0]: x={x:.2f}, y={y:.2f}, z={z:.2f}, doppler={d:.2f}")

        return points

    def parse_point_cloud(self, tlv_value_bytes):
        """Parse spherical-coordinate point cloud (TLV Type 1000)"""
        point_struct_format = '<ffff'  # 4 floats: range, azimuth, elevation, doppler
        point_size = struct.calcsize(point_struct_format)
        num_points = len(tlv_value_bytes) // point_size
        points = []
        for i in range(num_points):
            point_data = struct.unpack(point_struct_format, tlv_value_bytes[i*point_size : (i+1)*point_size])
            points.append(point_data)
        return points

    def parse_side_info(self, tlv_value_bytes):
        """Parse additional information for detected points (TLV Type 7)"""
        side_info_format = '<Hh'  # 2-byte SNR, 2-byte Noise
        side_info_size = struct.calcsize(side_info_format)
        num_points = len(tlv_value_bytes) // side_info_size
        side_info = []
        for i in range(num_points):
            info = struct.unpack(side_info_format, tlv_value_bytes[i*side_info_size : (i+1)*side_info_size])
            side_info.append(info) # (snr, noise)
        return side_info

    def parse_target_index(self, tlv_value_bytes):
        """Parse target index (TLV Type 1011)"""
        return list(tlv_value_bytes) # Returns a list of target IDs for each point

    def publish_point_cloud(self, frame_data):
        """Combines parsed data and publishes a PointCloud2 message."""
        points = frame_data.get('point_cloud')
        point_type = frame_data.get('point_type')
        
        if not points:
            return

        num_points = len(points)
        side_info = frame_data.get('side_info')
        target_indices = frame_data.get('target_indices')

        # Create a structured numpy array
        # We will convert spherical coordinates to cartesian (x,y,z)
        # and add other fields.
        cloud_data = np.zeros(num_points, dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('doppler', np.float32),
            ('snr', np.uint16),
            ('noise', np.int16),
            ('target_id', np.uint8)
        ])

        for i in range(num_points):
            if point_type == 'spherical':
                range_val, azimuth_rad, elevation_rad, doppler = points[i]
                
                # Spherical to Cartesian conversion
                cos_ele = np.cos(elevation_rad)
                sin_ele = np.sin(elevation_rad)
                cos_azi = np.cos(azimuth_rad)
                sin_azi = np.sin(azimuth_rad)
                
                # Correcting for inverted mount by negating X and Z axes
                cloud_data[i]['x'] = -1 * (range_val * cos_ele * sin_azi)
                cloud_data[i]['y'] = range_val * cos_ele * cos_azi
                cloud_data[i]['z'] = -1 * (range_val * sin_ele)
                cloud_data[i]['doppler'] = doppler
            elif point_type == 'cartesian':
                x, y, z, doppler = points[i]
                # Apply same coordinate corrections as spherical
                cloud_data[i]['x'] = -1 * x
                cloud_data[i]['y'] = y
                cloud_data[i]['z'] = -1 * z
                cloud_data[i]['doppler'] = doppler
            
            if side_info and i < len(side_info):
                cloud_data[i]['snr'] = side_info[i][0]
                cloud_data[i]['noise'] = side_info[i][1]
            
            if target_indices and i < len(target_indices):
                cloud_data[i]['target_id'] = target_indices[i]

        # Create PointCloud2 message
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id="ti_mmwave_0")
        
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='doppler', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='snr', offset=16, datatype=PointField.UINT16, count=1),
            PointField(name='noise', offset=18, datatype=PointField.INT16, count=1),
            PointField(name='target_id', offset=20, datatype=PointField.UINT8, count=1),
        ]
        
        point_cloud_msg = PointCloud2(
            header=header,
            height=1,
            width=num_points,
            is_dense=True,
            is_bigendian=False,
            fields=fields,
            point_step=21, # x,y,z,doppler (16) + snr,noise (4) + target_id (1) = 21
            row_step=21 * num_points,
            data=cloud_data.tobytes()
        )
        
        self.point_cloud_pub_.publish(point_cloud_msg)
        self.get_logger().info(f"Published point cloud with {num_points} points.")

    def publish_markers(self, track_array_msg):
        """Publishes a MarkerArray for visualization in RViz."""
        marker_array = MarkerArray()
        for track in track_array_msg.track:
            # Marker for the object itself (e.g., a sphere)
            marker = Marker()
            marker.header = track_array_msg.header
            marker.ns = "radar_tracks"
            marker.id = track.tid
            marker.type = Marker.CUBE # Changed from SPHERE to CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = track.posx
            marker.pose.position.y = track.posy
            marker.pose.position.z = track.posz
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.5  # Sphere diameter in meters
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.a = 0.8
            marker.color.r = 1.0
            marker.color.g = 0.1
            marker.color.b = 0.1
            marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
            marker_array.markers.append(marker)

            # Marker for the velocity vector
            vel_marker = Marker()
            vel_marker.header = track_array_msg.header
            vel_marker.ns = "velocity_vectors"
            vel_marker.id = track.tid
            vel_marker.type = Marker.ARROW
            vel_marker.action = Marker.ADD
            vel_marker.points.append(marker.pose.position) # Start of the arrow
            end_point = marker.pose.position
            end_point.x += track.velx * 0.5 # Scale velocity for visualization
            end_point.y += track.vely * 0.5
            vel_marker.points.append(end_point) # End of the arrow
            vel_marker.scale.x = 0.05  # Arrow shaft diameter
            vel_marker.scale.y = 0.1   # Arrow head diameter
            vel_marker.color.a = 0.8
            vel_marker.color.r = 0.0
            vel_marker.color.g = 1.0
            vel_marker.color.b = 0.0
            vel_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
            marker_array.markers.append(vel_marker)

            # Marker for the TID text
            text_marker = Marker()
            text_marker.header = track_array_msg.header
            text_marker.ns = "track_ids"
            text_marker.id = track.tid
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = track.posx
            text_marker.pose.position.y = track.posy
            text_marker.pose.position.z = track.posz + 0.5 # Display text above the marker cube
            text_marker.text = f"ID:{track.tid}"
            text_marker.scale.z = 0.4 # Text height
            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
            marker_array.markers.append(text_marker)

        self.marker_pub_.publish(marker_array)

    def shutdown(self):
        """Cleans up resources when the node is shut down."""
        self.get_logger().info("Shutting down node...")
        if self.cli_port and self.cli_port.is_open:
            try:
                self.get_logger().info("Sending 'sensorStop' command...")
                self.cli_port.write(b'sensorStop\n')
                time.sleep(0.2) 
            except Exception as e:
                self.get_logger().error(f"Failed to send sensorStop command: {e}")
            self.cli_port.close()
        if self.data_port and self.data_port.is_open:
            self.data_port.close()
        self.get_logger().info("All ports closed.")

def main(args=None):
    rclpy.init(args=args)
    driver_node = MobileTrackerDriver()
    try:
        driver_node.run()
    except KeyboardInterrupt:
        driver_node.get_logger().info("KeyboardInterrupt received, initiating shutdown...")
        driver_node.shutdown()  # Call shutdown here to ensure logging works
    finally:
        # The main shutdown logic is now handled in the except block.
        # This block ensures that the node is always destroyed and rclpy is shut down.
        if rclpy.ok():
            driver_node.destroy_node()
            rclpy.shutdown()
            driver_node.get_logger().info("RCLPY shut down successfully.")

if __name__ == '__main__':
    main()
