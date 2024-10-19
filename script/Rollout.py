#!/usr/bin/env python3
import rospy, message_filters, cv2, os, rospkg, torch
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import Image
from tf.transformations import quaternion_from_euler
from iL_network.FiveResNet18MLP5 import FiveResNet18MLP5
from cv_bridge import CvBridge
from torchvision import transforms

class Rollout:
        
    # Image Pre-Processor
    bridge = CvBridge()
    data_transforms = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __init__(self, model, weight_path, goal_images):
        rospy.init_node('rollout')
        self.pub_displacement = rospy.Publisher('spot/displacement', Pose, queue_size=10)

        # Setup Model
        self.model = model
        self.model.load_state_dict(torch.load(weight_path))
        self.model.cuda()
        self.model.eval()
        rospy.loginfo("Model Setup Complete")
        rospy.sleep(1.0)

        # Load Goal Images
        self.goal_images = goal_images

        # 5 images from SPOT
        front_left_camera_sub = message_filters.Subscriber('spot/front_left_camera/image_raw', Image)
        front_right_camera_sub = message_filters.Subscriber('spot/front_right_camera/image_raw', Image)
        back_camera_sub = message_filters.Subscriber('spot/back_camera/image_raw', Image)
        left_camera_sub = message_filters.Subscriber('spot/left_camera/image_raw', Image)
        right_camera_sub = message_filters.Subscriber('spot/right_camera/image_raw', Image)

        timeSynchronizer = message_filters.ApproximateTimeSynchronizer([front_left_camera_sub, front_right_camera_sub, back_camera_sub, left_camera_sub, right_camera_sub], 10, 0.1)
        timeSynchronizer.registerCallback(self.callback)
        rospy.loginfo("Init Complete")


    def callback(self, front_left_image, front_right_camera, back_camera, left_camera, right_camera):
        
        current_images = Rollout.tensor_from_image([front_left_image, front_right_camera, back_camera, left_camera, right_camera])

        predicted_angle = self.model(current_images, self.goal_images)

        displacement = Pose()
        displacement.position = Point(0, 0, 0)
        q = quaternion_from_euler(0, 0, predicted_angle)
        displacement.orientation = Quaternion(q[0], q[1], q[2], q[3])
        self.pub_displacement.publish(displacement)

    @staticmethod
    def tensor_from_image(images) -> torch.Tensor:

        tensor_images = []
        for img in images:

            cv2_image = Rollout.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')
            cv2_image = cv2.resize(cv2_image, (224, 224))
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)

            tensor_image = torch.tensor(cv2_image).permute(2, 0, 1).float().cuda() / 255.0
            tensor_image = Rollout.data_transforms(tensor_image)
            tensor_images.append(tensor_image)

        tensor_images = torch.stack(tensor_images).unsqueeze(0)

        return tensor_images

    # def apply_rotation_until_zero_yaw(self):
    #     step = 0
    #     while not rospy.is_shutdown():
    #         rospy.sleep(0.2)

    #         rospy.loginfo(f"Step N # {step}")
    #         if abs(self.current_yaw) < 0.01:
    #             rospy.loginfo("Yaw is approximately zero; stopping rotation.")
    #             break
    #         rospy.loginfo(f"Current Yaw is : {self.current_yaw} radians, which is {math.degrees(self.current_yaw):.2f} degrees")

    #         displacement = Pose()
    #         displacement.position = Point(0, 0, 0)
    #         q = quaternion_from_euler(0, 0, self.rotation_step)
    #         displacement.orientation = Quaternion(q[0], q[1], q[2], q[3])
    #         self.pub_displacement.publish(displacement)

    #         step += 1

if __name__ == '__main__':

    model = FiveResNet18MLP5()
    package_path = rospkg.RosPack().get_path('spot_IL_environment')
    weight_path = package_path + '/script/iL_network/weights/initial_0_133.pth'

    # Load goal images
    goal_images_path = package_path + '/script/iL_network/goal'
    goal_images = []
    for i in range(5):
            goal_image_path = os.path.join(goal_images_path, f"{i}.png")
            cv2_image = cv2.imread(goal_image_path)
            cv2_image = cv2.resize(cv2_image, (224, 224))
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
            tensor_image = torch.tensor(cv2_image).permute(2, 0, 1).float().cuda() / 255.0
            tensor_image = Rollout.data_transforms(tensor_image)

            goal_images.append(tensor_image)
    goal_images = torch.stack(goal_images).unsqueeze(0)

    rollout = Rollout(model, weight_path, goal_images)
    rospy.spin()
