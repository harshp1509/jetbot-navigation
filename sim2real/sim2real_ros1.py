import rospy
import numpy as np
from nav_msgs.msg import Odometry
import torch
from actor_critic import ActorCritic
import numpy
from geometry_msgs.msg import Twist

class Hound_RLHL_Control:
    def __init__(self):
        model_path = "/media/airangers/AiRangersData2/Harsh/model_1499.pt"

        self.state = np.zeros(7, dtype=np.float32)
        self.rate = 50
        self.next_move=False
        self.state[2:6] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.r=0.03  # wheel radius in meters
        self.l=0.10  #wheelbase in meters

        loaded_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.Model = ActorCritic()
        self.Model.load_state_dict(loaded_dict["model_state_dict"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Model.to(self.device)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)

    def main_loop(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            state = torch.Tensor(self.state).unsqueeze(0).to(self.device)
            if self.next_move:
                self.next_move=False
                with torch.no_grad():
                    # actions = self.Model.act_inference(state).squeeze(0).numpy().astype(numpy.float32)
                    actions = self.Model.act_inference(state).squeeze(0).cpu().numpy().astype(numpy.float32)
                    clipped_actions = numpy.clip(actions, -1, 1)
                ctrl = clipped_actions
                # ctrl = actions
                print("ctrl", ctrl)
                vel_right = ctrl[0]
                vel_left = ctrl[1]
                linear_x = (self.r/2.0) *(vel_right + vel_left)
                angular_z = (self.r/self.l) * (vel_right - vel_left)


                twist=Twist()
                twist.linear.x = linear_x
                twist.angular.z = angular_z
                pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
                pub.publish(twist)
                print("data published:", twist)
                rate.sleep()
                
    def odom_callback(self, odom):
        rb_x = odom.pose.pose.position.x
        rb_y = odom.pose.pose.position.y
        rb_z = odom.pose.pose.position.z

        self.state[0] = rb_x
        self.state[1] = rb_y
        self.state[2] = rb_z
        self.next_move=True

if __name__ == "__main__":
    rospy.init_node("h1_controller")
    planner = Hound_RLHL_Control()
    planner.main_loop()
    rospy.spin()
