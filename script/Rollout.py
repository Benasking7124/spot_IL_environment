#!/usr/bin/env python3
import rospy, message_filters, cv2, os, rospkg, torch, importlib
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

    TOLERANCE = 5e-3
    COMPELTED_COUNT = 5e1

    def __init__(self, model, weight_path, goal_images):
        self.pub_displacement = rospy.Publisher('spot/displacement', Pose, queue_size=10)
        rospy.set_param('spot/mission_completed', False)

        # Setup Model
        self.model = model
        self.model.load_state_dict(torch.load(weight_path, weights_only=True))
        self.model.cuda()
        self.model.eval()
        self.index = 0
        self.completed_cout = 0
        rospy.loginfo("IL Model Setup Complete")
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
        
        current_images = Rollout.tensor_from_images([front_left_image, front_right_camera, back_camera, left_camera, right_camera])

        with torch.no_grad():
            predicted_angle = self.model(current_images, self.goal_images)

        if abs(predicted_angle) < Rollout.TOLERANCE:
             
            self.completed_cout += 1

            if self.completed_cout > Rollout.COMPELTED_COUNT:

                self.completed_cout = 0
                self.index = 0
                rospy.loginfo('Mission Completed !')
                rospy.set_param('spot/mission_completed', True)
                return

        self.index += 1
        # rospy.loginfo(f'{self.index}: {predicted_angle}')

        displacement = Pose()
        displacement.position = Point(0, 0, 0)
        q = quaternion_from_euler(0, 0, predicted_angle)
        displacement.orientation = Quaternion(q[0], q[1], q[2], q[3])
        self.pub_displacement.publish(displacement)

    @staticmethod
    def tensor_from_images(images) -> torch.Tensor:

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

if __name__ == '__main__':

    rospy.init_node('rollout')

    # Get Model Parameters
    model_param = rospy.get_param('model_param', ['FiveResNet18MLP5', 'initial_0_133.pth'])
    model_name = model_param[0]
    model_module_name = 'iL_network.' + model_name
    weight_name = model_param[1]

    # Setup Model
    module = importlib.import_module(model_module_name)
    model = getattr(module, model_name)()

    # Setup Weight
    package_path = rospkg.RosPack().get_path('spot_IL_environment')
    weight_directory = package_path + '/script/iL_network/weights/'
    weight_path = weight_directory + weight_name
    rospy.loginfo(f'Model: {model_name}, Weight: {weight_name}')

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
