#! /usr/bin/env python3
import rospy, message_filters, os, cv2, rospkg
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point, Quaternion
from spot_IL.msg import StringStamped
from cv_bridge import CvBridge
from tf.transformations import quaternion_from_euler
import numpy as np

POS_STEP = 0.1
ROT_STEP = 0.017
LABEL_FILE_NAME = 'labels'

class RecordDemo:

    def __init__(self) -> None:
        rospy.init_node('record_demo')
        self.index = 0
        self.DATASET_PATH = rospkg.RosPack().get_path('spot_IL_environment') + '/dataset/'

        # Publish Control Command
        self.pub_displacement = rospy.Publisher('spot/displacement', Pose, queue_size=10)
        self.trajectory = np.empty([0, 7])

        # Record 5 images from SPOT
        front_left_camera_sub = message_filters.Subscriber('spot/front_left_camera/image_raw', Image)
        front_right_camera_sub = message_filters.Subscriber('spot/front_right_camera/image_raw', Image)
        back_camera_sub = message_filters.Subscriber('spot/back_camera/image_raw', Image)
        left_camera_sub = message_filters.Subscriber('spot/left_camera/image_raw', Image)
        right_camera_sub = message_filters.Subscriber('spot/right_camera/image_raw', Image)

        # Record action
        action_sub = message_filters.Subscriber('spot/action', StringStamped)

        # Record SPOT pose

        # Subscribe all topic together
        timeSynchronizer = message_filters.ApproximateTimeSynchronizer([front_left_camera_sub, front_right_camera_sub, back_camera_sub, left_camera_sub, right_camera_sub, action_sub], 10, 0.1)
        timeSynchronizer.registerCallback(self.callback)

    def callback(self, front_left_image, front_right_camera, back_camera, left_camera, right_camera, action):

        self.save_images([front_left_image, front_right_camera, back_camera, left_camera, right_camera])

        # Control Spot
        displacement = Pose()
        displacement.position = Point(0, 0, 0)
        displacement.orientation = Quaternion(0, 0, 0, 1)

        if action.string == 'w':
            print(self.index, 'forward')
            displacement.position.x = POS_STEP

        elif action.string == 's':
            print(self.index, 'backward')
            displacement.position.x = -POS_STEP

        elif action.string == 'a':
            print(self.index, 'left')
            displacement.position.y = POS_STEP

        elif action.string == 'd':
            print(self.index, 'right')
            displacement.position.y = -POS_STEP

        elif action.string == 'q':
            print(self.index, 'turn left')
            q = quaternion_from_euler(0, 0, ROT_STEP)
            displacement.orientation.x = q[0]
            displacement.orientation.y = q[1]
            displacement.orientation.z = q[2]
            displacement.orientation.w = q[3]

        elif action.string == 'e':
            print(self.index, 'turn right')
            q = quaternion_from_euler(0, 0, -ROT_STEP)
            displacement.orientation.x = q[0]
            displacement.orientation.y = q[1]
            displacement.orientation.z = q[2]
            displacement.orientation.w = q[3]

        self.trajectory = np.vstack((self.trajectory, [displacement.position.x, displacement.position.y, displacement.position.z, displacement.orientation.x, displacement.orientation.y, displacement.orientation.z, displacement.orientation.w]))
        np.save(self.DATASET_PATH + LABEL_FILE_NAME, self.trajectory)
        self.pub_displacement.publish(displacement)
        self.index += 1

    def save_images(self, images):
        folder_name = self.DATASET_PATH + format(self.index, '05d') + '/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for i in range(len(images)):
            cv2_image = CvBridge().imgmsg_to_cv2(images[i], desired_encoding='bgr8')
            image_name = folder_name + str(i) + '.png'
            cv2.imwrite(image_name, cv2_image)

if __name__ == '__main__':
    demo_recorder = RecordDemo()
    rospy.spin()