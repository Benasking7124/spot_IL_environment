#! /usr/bin/env python3
import rospy, message_filters, os, cv2, rospkg
from sensor_msgs.msg import Image
from spot_IL.msg import StringStamped
from cv_bridge import CvBridge

class RecordDemo:

    def __init__(self) -> None:
        rospy.init_node('record_demo')
        self.index = 0
        self.DATASET_PATH = rospkg.RosPack().get_path('spot_IL') + '/dataset/'

        # Record 5 images from SPOT
        front_left_camera_sub = message_filters.Subscriber('front_left_camera/image_raw', Image)
        front_right_camera_sub = message_filters.Subscriber('front_right_camera/image_raw', Image)
        back_camera_sub = message_filters.Subscriber('back_camera/image_raw', Image)
        left_camera_sub = message_filters.Subscriber('left_camera/image_raw', Image)
        right_camera_sub = message_filters.Subscriber('right_camera/image_raw', Image)

        # Record action
        action_sub = message_filters.Subscriber('spot/action', StringStamped)

        # Record SPOT pose

        # Subscribe all topic together
        timeSynchronizer = message_filters.ApproximateTimeSynchronizer([front_left_camera_sub, front_right_camera_sub, back_camera_sub, left_camera_sub, right_camera_sub, action_sub], 10, 0.1)
        timeSynchronizer.registerCallback(self.callback)

    def callback(self, front_left_image, front_right_camera, back_camera, left_camera, right_camera, action):

        self.save_images([front_left_image, front_right_camera, back_camera, left_camera, right_camera])

        if action.string == 'w':
            print(self.index, 'forward')

        elif action.string == 's':
            print(self.index, 'backward')

        elif action.string == 'a':
            print(self.index, 'left')

        elif action.string == 'd':
            print(self.index, 'right')

        elif action.string == 'q':
            print(self.index, 'turn left')

        elif action.string == 'e':
            print(self.index, 'turn right')

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