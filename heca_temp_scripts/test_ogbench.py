from heca.environment.scenes.ogbench.scene import OGBenchScene
from heca.environment.scenes.scene import Scene

scene = Scene.load(OGBenchScene.Config())
scene.test()

# [
# 'proprio/joint_pos',
# 'proprio/joint_vel',
# 'proprio/effector_pos',
# 'proprio/effector_yaw',
# 'proprio/gripper_opening',
# 'proprio/gripper_vel',
# 'proprio/gripper_contact',
# 'privileged/block_0_pos',
# 'privileged/block_0_quat',
# 'privileged/block_0_yaw',
# 'privileged/button_0_state',
# 'privileged/button_0_pos',
# 'privileged/button_0_vel',
# 'privileged/button_1_state',
# 'privileged/button_1_pos',
# 'privileged/button_1_vel',
# 'privileged/drawer_pos',
# 'privileged/drawer_vel',
# 'privileged/drawer_handle_pos',
# 'privileged/drawer_handle_yaw',
# 'privileged/window_pos',
# 'privileged/window_vel',
# 'privileged/window_handle_pos',
# 'privileged/window_handle_yaw',
# 'prev_button_states',
# 'button_states',
# 'prev_qpos',
# 'prev_qvel',
# 'qpos',
# 'qvel',
# 'control',
# 'time',
# 'goal',
# 'goal_rendered'
# ]
