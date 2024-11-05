#!/usr/bin/env python3
import rospy, subprocess, rospkg, csv
import numpy as np
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState

ERROR_TOLERANCE = 1e-2

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
                rospy.loginfo("Gazebo model pose set successfully!")
            else:
                rospy.logwarn("Failed to set Gazebo model pose.")
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

                rospy.loginfo(completed_state)
                for i in range(len(goal_state)):
                    error += abs(completed_state[i] - goal_state[i])

            else:
                rospy.logwarn("Failed to get model state.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

        rospy.loginfo(error)
        if error < ERROR_TOLERANCE:
            success_cout += 1
            
    success_rate = round((success_cout / len(initial_states)) * 100, 2)
    rospy.loginfo(f'Success Rate: {success_rate}')
    return success_rate

if __name__ == '__main__':

    # ROS Initialization
    rospy.init_node('evaluation')

    rospy.wait_for_service('/gazebo/set_model_state')
    set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    rospy.wait_for_service('/gazebo/get_model_state')
    get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    rospy.set_param('spot/mission_completed', False)

    # Model
    test_model = [['FiveResNet18MLP5', 'initial_0_133.pth'],
                  ['FiveResNet18MLP5', 'mixed_2_311.pth'],
                  ['FiveResNet18MLP5', 'mixed_0_297.pth']]

    # Goal State
    goal_state = [0, 0, 0.6885, 0, 0, 0, 1]

    # Initial State
    initial_angle = np.array([170, 135, 90, 60, 30, 0, -30, -60, -90, -135, -170])
    initial_angle = (initial_angle / 180) * np.pi
    initial_quaternion = []
    for angle in initial_angle:
        q = quaternion_from_euler(0, 0, angle)
        initial_quaternion.append(q)

    initial_state = [[0, 0, 0.68, initial_quaternion[0][0], initial_quaternion[0][1], initial_quaternion[0][2], initial_quaternion[0][3]],
                     [0, 0, 0.68, initial_quaternion[1][0], initial_quaternion[1][1], initial_quaternion[1][2], initial_quaternion[1][3]],
                     [0, 0, 0.68, initial_quaternion[2][0], initial_quaternion[2][1], initial_quaternion[2][2], initial_quaternion[2][3]],
                     [0, 0, 0.68, initial_quaternion[3][0], initial_quaternion[3][1], initial_quaternion[3][2], initial_quaternion[3][3]],
                     [0, 0, 0.68, initial_quaternion[4][0], initial_quaternion[4][1], initial_quaternion[4][2], initial_quaternion[4][3]],
                     [0, 0, 0.68, initial_quaternion[5][0], initial_quaternion[5][1], initial_quaternion[5][2], initial_quaternion[5][3]],
                     [0, 0, 0.68, initial_quaternion[6][0], initial_quaternion[6][1], initial_quaternion[6][2], initial_quaternion[6][3]],
                     [0, 0, 0.68, initial_quaternion[7][0], initial_quaternion[7][1], initial_quaternion[7][2], initial_quaternion[7][3]],
                     [0, 0, 0.68, initial_quaternion[8][0], initial_quaternion[8][1], initial_quaternion[8][2], initial_quaternion[8][3]],
                     [0, 0, 0.68, initial_quaternion[9][0], initial_quaternion[9][1], initial_quaternion[9][2], initial_quaternion[9][3]],
                     [0, 0, 0.68, initial_quaternion[10][0], initial_quaternion[10][1], initial_quaternion[10][2], initial_quaternion[10][3]]]

    # Record the result
    result = []
    package_path = rospkg.RosPack().get_path('spot_IL_environment')
    result_path = package_path + '/script/iL_network/evaluation.csv'

    # Original Environment
    env_status = 'Original'
    success_rates = [env_status]

    for model in test_model:
        rospy.set_param('model_param', model)
        subprocess.Popen(['rosrun', 'spot_IL_environment', 'Rollout.py'])
        
        success_rate = evaluate(initial_state, goal_state)
        success_rates.append(success_rate)

        subprocess.call(['rosnode', 'kill', 'rollout'])
        rospy.sleep(1)

    # Record Sucess Rate
    result.append(success_rates)

    # Write result to file
    with open(result_path, "w", newline="") as file:
        writer = csv.writer(file)
        
        column_headers =  [''] + ["_".join(item) for item in test_model]
        writer.writerow(column_headers)
        
        writer.writerows(result)