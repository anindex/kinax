from urdf_parser_py.urdf import URDF
from kinax.utils.files import get_robot_path


if __name__ == '__main__':
    model_path = get_robot_path() / 'franka_description' / 'robots' / 'panda_arm_no_gripper.urdf'
    robot = URDF.from_xml_file(model_path)
