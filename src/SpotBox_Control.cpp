#include <ros/ros.h>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <geometry_msgs/Pose.h>

class SpotBox_Control:public gazebo::ModelPlugin
{
private:
    gazebo::physics::ModelPtr model;
    ros::NodeHandle nh;
    ros::Subscriber pose_subscriber;

public:
    void Load(gazebo::physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
        ROS_INFO("Loading SpotBox Control Plugin\n");
        int argc = 0;
        char **argv = NULL;
        model = _model;

        // ROS Node and Subscriber
        ros::init(argc, argv, "spot_box_control");
        pose_subscriber = nh.subscribe("spot/displacement", 10, &SpotBox_Control::Callback, this);
        ROS_INFO("SpotBox Control initialized\n");
    }

    void Callback(const geometry_msgs::PoseConstPtr &msg)
    {
        ignition::math::Pose3d current_pose = this->model->WorldPose();
        ignition::math::Quaternion displacement_pos = ignition::math::Quaternion(0.0, msg->position.x, msg->position.y, msg->position.z);

        // Position Displacement
        displacement_pos = current_pose.Rot() * displacement_pos * current_pose.Rot().Inverse();
        current_pose.Pos().X() += displacement_pos.X();
        current_pose.Pos().Y() += displacement_pos.Y();
        current_pose.Pos().Z() += displacement_pos.Z();

        // Angular Displacement
        current_pose.Rot() *= ignition::math::Quaternion(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);

        model->SetWorldPose(current_pose);
    }
};

GZ_REGISTER_MODEL_PLUGIN(SpotBox_Control)