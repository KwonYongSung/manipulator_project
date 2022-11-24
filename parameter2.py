import rclpy
import rclpy.node
#from rcl_interfaces.srv import GetParameters
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.msg import SetParametersResult
from . import teleop_marker

class MinimalParam(rclpy.node.Node):
    def __init__(self):
        super().__init__('minimal_param_node')
        
        self.declare_parameter("turtle_param", "False") #터틀봇의 parameter
        timer_period = 2  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        param = self.get_parameter('turtle_param').get_parameter_value().string_value 
        #파라미터 값이 True면 터틀봇이 동작하게 하기
        
        self.get_logger().info('turtle %s!' % param)
        
        

def main():
    rclpy.init()
    node = MinimalParam()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
