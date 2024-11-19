#!/usr/bin/env python3
import rospy, rospkg, yaml
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
from std_srvs.srv import Empty

def spawn_gazebo_model(gazebo_model):
    
    ## Get the path for the Gazebo model
    package_path = rospkg.RosPack().get_path('spot_IL_environment')
    gazebo_model_path = package_path + '/models/' + gazebo_model['type']

    ## Load the model file
    with open(gazebo_model_path, "r") as gazebo_model_file:
        gazebo_model_xml = gazebo_model_file.read()
    
    ## Get the pose of Gazebo model
    gazebo_model_pose = Pose()
    gazebo_model_pose.position.x = gazebo_model['pose'][0]
    gazebo_model_pose.position.y = gazebo_model['pose'][1]
    gazebo_model_pose.position.z = gazebo_model['pose'][2]

    q = quaternion_from_euler(gazebo_model['pose'][3], gazebo_model['pose'][4], gazebo_model['pose'][5])
    gazebo_model_pose.orientation.x = q[0]
    gazebo_model_pose.orientation.y = q[1]
    gazebo_model_pose.orientation.z = q[2]
    gazebo_model_pose.orientation.w = q[3]

    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    ## Spawn the Gazebo model
    try:
        spawn_model(gazebo_model['name'], gazebo_model_xml, '', gazebo_model_pose, 'world')
    
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to spawn model {gazebo_model['name']}: {e}")

if __name__ == '__main__':

    rospy.init_node('spawn_world')

    package_path = rospkg.RosPack().get_path('spot_IL_environment')
    world_name = rospy.get_param('world_name')
    world_path = package_path + '/worlds/' + world_name
    with open(world_path, "r") as f:
        world = yaml.safe_load(f)
    gazebo_model_list = world['models']

    for gazebo_model in gazebo_model_list:
        spawn_gazebo_model(gazebo_model)

    if rospy.get_param('gazebo_headless') is True:

        rospy.wait_for_service('/gazebo/unpause_physics')
        unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        
        try:
            unpause_physics()
            rospy.loginfo("Gazebo unpaused successfully.")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)