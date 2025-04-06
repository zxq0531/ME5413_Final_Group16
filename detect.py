#!/usr/bin/env python3

# Create a node to subscribe to the topic from the camera
# Process the image with ocr and get the result bounding boxes
# Calculate the angle of the object with the parameters of the camera
# Publish the result to new topic

import time

import cv2
import easyocr
import ipdb
import numpy as np
import rospy
import tf
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion

# import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"==>> device: {device}")


class Visual:
    def __init__(self, rate=10):
        rospy.loginfo(f"running rate: {rate}")
        self.detect_mode = "number"
        self.bridge = CvBridge()
        self.rate = rate
        self.my_array=[0] * 10

        
        self.img_curr = None
        self.img_curr_gray = None
        self.num_detect_result = [0] * 10  # 1 for detected, 0 for not detected
        self.camera_info = rospy.wait_for_message("/front/camera_info", CameraInfo)
        self.intrinsic = np.array(self.camera_info.K).reshape(3, 3)
        self.projection = np.array(self.camera_info.P).reshape(3, 4)
        self.distortion = np.array(self.camera_info.D)
        self.img_frame = self.camera_info.header.frame_id
        self.ocr_detector = easyocr.Reader(["en"], gpu=True)
        self.numberposelists = np.zeros((10, 2))
        self.curr_odom = None

        self.tf_sub = tf.TransformListener()
        self.img_sub = rospy.Subscriber("/front/image_raw", Image, self.img_callback)
        rospy.loginfo("visual node initialized")

    def odom_callback(self, msg):
        self.curr_odom = msg

    def img_callback(self, msg: Image):
        self.img_curr = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def detect_mode_callback(self, msg):
        self.detect_mode = msg.data

    def map_callback(self, msg):
        data = list(msg.data)
        for y in range(msg.info.height):
            for x in range(msg.info.width):
                i = x + (msg.info.height - 1 - y) * msg.info.width
                if data[i] >= 75:
                    data[i] = 100
                elif (data[i] >= 0) and (data[i] < 50):  # free
                    data[i] = 0
                else:  # unknown
                    data[i] = -1
        self.map = np.array(data).reshape(msg.info.height, msg.info.width)

    def goal_callback(self, msg):
        if msg.data[:4] == "/box":
            self.noi = msg.data[-1]
            rospy.loginfo(f"number of interest: {self.noi}")

    def scan_callback(self, msg: LaserScan):
        self.scan_curr = msg.ranges
        self.scan_params_curr = [msg.angle_min, msg.angle_max, msg.angle_increment]

    def run(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if self.img_curr is None:
                continue
            rospy.loginfo_throttle(2, f"detect_mode: {self.detect_mode}")

            if self.detect_mode == "number":
                if self.img_curr is None:
                    continue
                # img_gray = cv2.cvtColor(self.img_curr, cv2.COLOR_BGR2GRAY)
                # img_bin = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)[1]
                # cv2.imshow("img_bin", img_bin)
                # cv2.waitKey(1)
                result = self.ocr_detector.readtext(self.img_curr, batch_size=2, allowlist="0123456789")
                img_show = self.img_curr.copy()
                for detection in result:
                    # detection[0]: the bounding box of the detected text
                    # detection[1]: the detected text
                    # detection[2]: the confidence of the detected text
                    if len(detection[1]) > 1:  # not a single digit
                        continue
                    diag_vec = np.array(detection[0][2]) - np.array(detection[0][0])
                    diag_len = np.linalg.norm(diag_vec)
                    if 1:
                        cv2.rectangle(
                            img_show,
                            (int(detection[0][0][0]), int(detection[0][0][1])),
                            (int(detection[0][2][0]), int(detection[0][2][1])),
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            img_show,
                            detection[1] + f" {detection[2]:.2f}",
                            (int(detection[0][0][0]), int(detection[0][0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow("img", img_show)
                        cv2.waitKey(1)
                    
                    if detection[2] < 0.99:
                        continue
                    if (
                        diag_len < 60
                    ):  # prevent the case that  Recognizing 1 too early leads to incorrect distance estimation
                        continue
                    self.my_array[detection[1]]+=1
                    
            min_count = min(self.my_array)
            min_numbers = [i for i, cnt in enumerate(self.my_array) if cnt == min_count]
            print("出现次数最少的数字是:", min_numbers)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("visual")
    v = Visual(rate=30)
    v.run()