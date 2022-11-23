import rclpy
import rclpy.node
from rcl_interfaces.srv import GetParameters
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterType
from . import teleop_marker

class MinimalParam(rclpy.node.Node):
    def __init__(self):
        super().__init__('minimal_param_node')
        #self.tele.set_parameters([rclpy.parameter.Parameter('ArucoNode','manipulator1_param',rclpy.Parameter.Type.STRING,'*****')])
        timer_period = 2  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        self.tele = teleop_marker.ArucoNode()
        my_param = self.tele.get_parameter('manipulator1_param').get_parameter_value().string_value
        print('%s' % my_param)
        #my_param = rclpy.node.Node.get_parameter('manipulator1_param')
         #teleop_marker.ArucoNode.set_parameters([rclpy.parameter.Parameter('ArucoNode','manipulator1_param',rclpy.Parameter.Type.STRING,'True')])
        #print('%s' % my_param)

def main():
    rclpy.init()
    node = MinimalParam()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
