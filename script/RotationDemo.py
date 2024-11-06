#! /usr/bin/env python3 
import rospy, message_filters, os, cv2, rospkg
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point, Quaternion
from spot_IL.msg import StringStamped
from cv_bridge import CvBridge
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from gazebo_msgs.msg import ModelStates
import numpy as np
import math

POS_STEP = 0.1
ROT_STEP = 0.017
LABEL_FILE_NAME = 'labels'

class RecordDemo:

    def __init__(self) -> None:
        rospy.init_node('record_demo', anonymous=True)
        rospy.loginfo("Initializing RecordDemo Node...")
        rospy.sleep(5)  
        
        self.index = 0
        self.DATASET_PATH = rospkg.RosPack().get_path('spot_IL') + '/dataset/'

        # Publish Control Command
        self.pub_displacement = rospy.Publisher('spot/displacement', Pose, queue_size=10)
        self.trajectory = np.empty([0, 2])

        # Record 5 images from SPOT
        front_left_camera_sub = message_filters.Subscriber('spot/front_left_camera/image_raw', Image)
        front_right_camera_sub = message_filters.Subscriber('spot/front_right_camera/image_raw', Image)
        back_camera_sub = message_filters.Subscriber('spot/back_camera/image_raw', Image)
        left_camera_sub = message_filters.Subscriber('spot/left_camera/image_raw', Image)
        right_camera_sub = message_filters.Subscriber('spot/right_camera/image_raw', Image)

        # Subscribe to model states to get the current yaw and position
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)

        # Subscribe all topic together
        self.timeSynchronizer = message_filters.ApproximateTimeSynchronizer(
            [front_left_camera_sub, front_right_camera_sub, back_camera_sub, left_camera_sub, right_camera_sub],
            30,  # Increased queue size
            0.5  # Increased synchronization threshold (more lenient)
        )
        self.timeSynchronizer.registerCallback(self.turn_callback)

        self.current_yaw = 0.0
        self.current_position_x = 0.0
        self.current_position_y = 0.0
        self.left_turn_complete = False

    def model_states_callback(self, msg):
        try:
            robot_index = msg.name.index('spot')
            orientation = msg.pose[robot_index].orientation
            _, _, self.current_yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
            self.current_position_x = msg.pose[robot_index].position.x
            self.current_position_y = msg.pose[robot_index].position.y
        except ValueError:
            rospy.logwarn("Spot robot not found in /gazebo/model_states")
    
    def turn_callback(self, front_left_image, front_right_image, back_image, left_image, right_image):
        if not self.left_turn_complete:
            self.turn_left(front_left_image, front_right_image, back_image, left_image, right_image)
        else:
            self.turn_right(front_left_image, front_right_image, back_image, left_image, right_image)
            
    def turn_left(self, front_left_image, front_right_camera, back_camera, left_camera, right_camera):
        if self.current_position_x > -0.2:
            if abs(self.current_yaw) >= 0.018*160:
                rospy.loginfo(f"Turn Left -> Rotation demo N# {self.index}")
                rospy.loginfo(f"Current Yaw : {math.degrees(self.current_yaw):.2f} degrees")

                displacement = Pose()
                displacement.position = Point(0, 0, 0)
                displacement.orientation = Quaternion(0, 0, 0, 1)

                q = quaternion_from_euler(0, 0, ROT_STEP)
                displacement.orientation = Quaternion(q[0], q[1], q[2], q[3])

                # Publish displacement
                self.pub_displacement.publish(displacement)
                rospy.sleep(0.5)  

                # Save images
                self.save_images([front_left_image, front_right_camera, back_camera, left_camera, right_camera])

                # Save trajectory
                self.trajectory = np.vstack((self.trajectory, [self.current_position_x, self.current_yaw]))
                np.save(self.DATASET_PATH + LABEL_FILE_NAME, self.trajectory)

                self.index += 1
            else : 
                rospy.loginfo("###################################################################")
                rospy.loginfo(f"Turn left -> Changing position")
                rospy.loginfo("###################################################################")

                displacement = Pose()
                displacement.position = Point(0, 0, 0)
                displacement.orientation = Quaternion(0, 0, 0, 1)

                displacement.position.x = POS_STEP
                self.pub_displacement.publish(displacement)

                rospy.sleep(0.5)


                displacement = Pose()
                displacement.position = Point(0, 0, 0)
                displacement.orientation = Quaternion(0, 0, 0, 1)

                q = quaternion_from_euler(0, 0, -3.124 - self.current_yaw )
                displacement.orientation = Quaternion(q[0], q[1], q[2], q[3])
                self.pub_displacement.publish(displacement)

                rospy.sleep(0.5)

                rospy.loginfo(f"Current X position: {self.current_position_x:.2f}")            
        else:
            rospy.loginfo("Left turn complete. Preparing to turn right.")
            self.left_turn_complete = True  # flag to switch to right turn
            rospy.sleep(5)
            rospy.loginfo("---------------------------------------------------------")
            rospy.loginfo("Starting right turn.")
            rospy.loginfo("---------------------------------------------------------")

    def turn_right(self, front_left_image, front_right_camera, back_camera, left_camera, right_camera):

        if self.current_position_x < 2:
            
            if abs(self.current_yaw) >= 0.018*160:
                rospy.loginfo(f"Turn Right -> Rotation demo N# {self.index}")
                rospy.loginfo(f"Current Yaw : {math.degrees(self.current_yaw):.2f} degrees")

                displacement = Pose()
                displacement.position = Point(0, 0, 0)
                displacement.orientation = Quaternion(0, 0, 0, 1)

                q = quaternion_from_euler(0, 0, -ROT_STEP)
                displacement.orientation = Quaternion(q[0], q[1], q[2], q[3])

                # Publish displacement
                self.pub_displacement.publish(displacement)
                rospy.sleep(0.5)  

                # Save images
                self.save_images([front_left_image, front_right_camera, back_camera, left_camera, right_camera])

                # Save trajectory
                self.trajectory = np.vstack((self.trajectory, [self.current_position_x, self.current_yaw]))
                np.save(self.DATASET_PATH + LABEL_FILE_NAME, self.trajectory)

                self.index += 1
            else : 
                rospy.loginfo("###################################################################")
                rospy.loginfo(f"Turning Right -> Changing position ")
                rospy.loginfo("###################################################################")

                displacement = Pose()
                displacement.position = Point(0, 0, 0)
                displacement.orientation = Quaternion(0, 0, 0, 1)

                displacement.position.x = -POS_STEP
                self.pub_displacement.publish(displacement)

                rospy.sleep(0.5)


                displacement = Pose()
                displacement.position = Point(0, 0, 0)
                displacement.orientation = Quaternion(0, 0, 0, 1)

                q = quaternion_from_euler(0, 0, 3.124 - self.current_yaw )
                displacement.orientation = Quaternion(q[0], q[1], q[2], q[3])
                self.pub_displacement.publish(displacement)

                rospy.sleep(0.5)

                rospy.loginfo(f"Current X position: {self.current_position_x:.2f}")  
        else : 
            return

        
    def save_images(self, images):
        folder_name = self.DATASET_PATH + format(self.index, '05d') + '/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for i in range(len(images)):
            try:
                cv2_image = CvBridge().imgmsg_to_cv2(images[i], desired_encoding='bgr8')
                image_name = folder_name + str(i) + '.png'
                cv2.imwrite(image_name, cv2_image)
            except Exception as e:
                rospy.logerr(f"Failed to save image: {e}")

if __name__ == '__main__':
    rospy.loginfo("Starting AutoDemoCollector Node...")
    demo_recorder = RecordDemo()
    rospy.spin()
