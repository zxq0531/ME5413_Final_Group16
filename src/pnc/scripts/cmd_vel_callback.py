import rospy
from geometry_msgs.msg import Twist

def cmd_vel_callback(msg):
    # Print out the published Twist values
    rospy.loginfo("Published cmd_vel: linear.x = %.2f, angular.z = %.2f", msg.linear.x, msg.angular.z)

if __name__ == '__main__':
    rospy.init_node('cmd_vel_logger')
    # Subscribe to /cmd_vel topic
    cmd_vel_sub = rospy.Subscriber("/cmd_vel", Twist, cmd_vel_callback)
    rospy.spin()
