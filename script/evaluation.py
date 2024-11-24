#!/usr/bin/env python3
import rospy, subprocess, rospkg, csv, yaml, os, xmltodict
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState, DeleteModel, SpawnModel

ERROR_TOLERANCE = 5e-2

def evaluate_IL_models(models, env_status, initial_states, goal_state):

    success_rates = [env_status]

    for model in models:
        rospy.set_param('model_param', model)
        subprocess.Popen(['rosrun', 'spot_IL_environment', 'Rollout.py'])
        
        success_rate = evaluate(initial_states, goal_state)
        success_rates.append(success_rate)

        subprocess.call(['rosnode', 'kill', 'rollout'])
        rospy.sleep(1)

    return success_rates

def evaluate(initial_states, goal_state):

    success_cout = 0

    for state in initial_states:

        # Set New Model Pose
        pose = Pose()
        pose.position.x = state[0]
        pose.position.y = state[1]
        pose.position.z = state[2]
        pose.orientation.x = state[3]
        pose.orientation.y = state[4]
        pose.orientation.z = state[5]
        pose.orientation.w = state[6]

        model_state = ModelState()
        model_state.model_name = 'spot'
        model_state.pose = pose

        try:
            response = set_model_state(model_state)
            if response.success:
                rospy.loginfo("Spot initial pose set successfully!")
            else:
                rospy.logwarn("Failed to set Spot initial pose.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

        rospy.set_param('spot/mission_completed', False)
        
        # Wait for Completion
        while rospy.get_param('spot/mission_completed') != True:
            pass

        # Check Success
        error = 0
        try:
            response = get_model_state(model_state.model_name, '')
            
            if response.success:
                completed_state = [response.pose.position.x, response.pose.position.y, response.pose.position.z, response.pose.orientation.x, response.pose.orientation.y, response.pose.orientation.z, response.pose.orientation.w]

                for i in range(len(goal_state)):
                    error += abs(completed_state[i] - goal_state[i])

            else:
                rospy.logwarn("Failed to get model state.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

        rospy.loginfo(f'Prediction Difference: {error}')
        if error < ERROR_TOLERANCE:
            success_cout += 1
            
    success_rate = round((success_cout / len(initial_states)) * 100, 2)
    rospy.loginfo(f'Success Rate: {success_rate}')
    return success_rate

def record_result(result_path, result, header_list):

    with open(result_path, "w", newline="") as file:
        writer = csv.writer(file)
        
        column_headers =  [''] + ["_".join(item) for item in header_list]
        writer.writerow(column_headers)
        
        writer.writerows(result)

def spawn_gazebo_model(gazebo_model, sdf_version):
    
    ## Get the path for the Gazebo model
    package_path = rospkg.RosPack().get_path('spot_IL_environment')
    gazebo_model_type = gazebo_model['uri'].split("://", 1)[1]
    gazebo_model_path = package_path + '/models/' + gazebo_model_type

    if not os.path.exists(gazebo_model_path):
        gazebo_model_path = os.path.expanduser('~/.gazebo/models/' + gazebo_model_type)

    ## Load the model file
    with open(gazebo_model_path + '/model.config') as f:
            gazebo_model_config = f.read()

    gazebo_model_sdfs = xmltodict.parse(gazebo_model_config)['model']['sdf']
    
    if isinstance(gazebo_model_sdfs, list):
         gazebo_model_sdf = next((item for item in gazebo_model_sdfs if item['@version'] == sdf_version), gazebo_model_sdfs[-1])['#text']

    else:
         gazebo_model_sdf = gazebo_model_sdfs['#text']

    gazebo_model_sdf_path = gazebo_model_path + '/' + gazebo_model_sdf

    with open(gazebo_model_sdf_path, "r") as f:
        gazebo_model_xml = f.read()
    
    ## Get the pose of Gazebo model
    pose_list = [float(num) for num in gazebo_model['pose'].split()]
    gazebo_model_pose = Pose()
    gazebo_model_pose.position.x = pose_list[0]
    gazebo_model_pose.position.y = pose_list[1]
    gazebo_model_pose.position.z = pose_list[2]

    q = quaternion_from_euler(pose_list[3], pose_list[4], pose_list[5])
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

    # ----- ROS Initialization -----
    rospy.init_node('evaluation')

    rospy.wait_for_service('/gazebo/set_model_state')
    set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    rospy.wait_for_service('/gazebo/get_model_state')
    get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    rospy.wait_for_service('/gazebo/delete_model')
    delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

    rospy.set_param('spot/mission_completed', False)

    # ----- IL Model -----
    test_IL_model = [['FiveResNet18MLP5', 'initial_0_133.pth'],
                     ['FiveResNet18MLP5', 'mixed_2_311.pth'],
                     ['FiveResNet18MLP5', 'mixed_0_297.pth']]

    # ----- Goal State -----
    goal_state = [0, 0, 0.6885, 0, 0, 0, 1]

    # ----- Initial State -----
    package_path = rospkg.RosPack().get_path('spot_IL_environment')
    initial_states_path = package_path + '/script/iL_network/evaluation/initial_states.yaml'
    with open(initial_states_path, "r") as f:
        initial_states = yaml.safe_load(f)

    # ----- Setup for recording the result -----
    result = []
    result_path = package_path + '/script/iL_network/evaluation/evaluation.csv'

    # ----- Init Complete -----
    rospy.loginfo('Evaluation Init Completed')

    # ----- Original Environment -----
    env_status = 'Original'
    success_rates = evaluate_IL_models(test_IL_model, env_status, initial_states, goal_state)
    result.append(success_rates)
    record_result(result_path, result, test_IL_model)

    # ----- Remove one model at a time -----
    ## Get Gazebo model list
    world_name = rospy.get_param('world_name')
    world_path = package_path + '/worlds/' + world_name
    with open(world_path, "r") as f:
        world_xml = f.read()
    sdf_version = xmltodict.parse(world_xml)['sdf']['@version']
    gazebo_models_xml = xmltodict.parse(world_xml)['sdf']['world']['include']

    for gazebo_model in gazebo_models_xml:

        if 'wall' in gazebo_model['name']:
            continue

        # Remove Gazebo model
        delete_model(gazebo_model['name'])

        # Evaluate IL models
        env_status = f"Remove {gazebo_model['name']}"
        success_rates = evaluate_IL_models(test_IL_model, env_status, initial_states, goal_state)
        result.append(success_rates)
        record_result(result_path, result, test_IL_model)

        # Add Gazebo model back
        spawn_gazebo_model(gazebo_model, sdf_version)

    # ----- Write result to file -----
    record_result(result_path, result, test_IL_model)
