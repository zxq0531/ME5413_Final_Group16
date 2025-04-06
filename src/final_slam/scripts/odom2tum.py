import rosbag
import tf

def odom_to_tum_format(bag_file, topic, output_file):
    bag = rosbag.Bag(bag_file)
    with open(output_file, 'w') as f_out:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            time = t.to_sec()
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            f_out.write(f"{time} {pos.x} {pos.y} {pos.z} {ori.x} {ori.y} {ori.z} {ori.w}\n")
    bag.close()

def odom_to_tum_format_transformed(bag_file, topic, output_file):
    bag = rosbag.Bag(bag_file)
    with open(output_file, 'w') as f_out:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            time = t.to_sec()
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation

            # Rotate counterclockwise 90 degrees around the Z-axis
            x_new = -pos.y
            y_new = pos.x
            z_new = pos.z + 2.66

            # Original quaternion
            quat_orig = [ori.x, ori.y, ori.z, ori.w]
            # Quaternion representing 90-degree counterclockwise rotation around Z-axis
            q_rot = tf.transformations.quaternion_from_euler(0, 0, 1.5708)
            # New quaternion = rotation * original
            quat_new = tf.transformations.quaternion_multiply(q_rot, quat_orig)

            f_out.write(f"{time} {x_new} {y_new} {z_new} {quat_new[0]} {quat_new[1]} {quat_new[2]} {quat_new[3]}\n")

    bag.close()

bagfile = '/home/oreo/final/ME5413_Final_Group16/src/final_slam_my/bagfiles/robot_run_1.bag'

odom_to_tum_format_transformed(bagfile, '/odometry/filtered', 'odom_filtered_1.txt')
odom_to_tum_format(bagfile, '/gazebo/ground_truth/state', 'ground_truth_1.txt')
