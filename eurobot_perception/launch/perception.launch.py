from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    crate_node = Node(
        package='eurobot_perception',
        executable='crate_perception',
        name='crate_perception'
    )

    pantry_node = Node(
        package='eurobot_perception',
        executable='pantry_detection',
        name='pantry_detection'
    )

    # Pantry node will start after crate node
    return LaunchDescription([crate_node, pantry_node])
