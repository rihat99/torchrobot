class RobotData:
    """
    Data class for storing the dynamic state of the robot.
    
    Attributes:
      - joint_values: dictionary mapping joint id to its configuration.
          For spherical joints: {'q': tensor(4)}
          For SE3 joints: {'translation': tensor(3), 'q': tensor(4)}
      - joint_transforms: dictionary mapping joint id to computed 4x4 transforms.
      - body_transforms: dictionary mapping body name to computed 4x4 transforms.
    """
    def __init__(self):
        self.joint_values = {}
        self.joint_transforms = {}
        self.body_transforms = {}