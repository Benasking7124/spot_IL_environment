#! /usr/bin/env python3
import rospy, sys, termios, tty
from spot_IL.msg import StringStamped

if __name__ == '__main__':
    rospy.init_node('keyboard_control')
    pub = rospy.Publisher('/spot/action', StringStamped, queue_size=10)

    fd = sys.stdin.fileno()
    old_setting = termios.tcgetattr(fd)
    tty.setraw(sys.stdin.fileno())

    output = StringStamped()

    try:
        while not rospy.is_shutdown():
            input = sys.stdin.read(1)

            if input == 'x':
                break
                
            output.string = input
            output.header.stamp = rospy.get_rostime()
            # output.header.stamp = rospy.Time(secs=68, nsecs=267000000)
            pub.publish(output)
    
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_setting)
