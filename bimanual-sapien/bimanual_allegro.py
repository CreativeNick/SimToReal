import sapien
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
import torch


# 828
left_arm_init_qpos = np.array(
    [
        -4.6746,
        -0.6805,
        1.5093,
        -2.3778,
        -0.8824,
        -0.0632,
    ]
)

# 830
right_arm_init_qpos = np.array(
    [
        -1.5676,
        -2.4176,
        -1.4704,
        -0.8341,
        0.89473,
        0.08133,
    ]
)

left_hand_init_qpos = np.zeros(16)

right_hand_init_qpos = np.zeros(16)


@register_agent()
class Bimanual_Allegro(BaseAgent):
    uid = "Bimanual_Allegro"
    urdf_path = "assets/urdf/ur5e_allegro/robots/dual_ur5e_allegro_inertia_changed.urdf"
    #srdf_path = "assets/urdf/ur5e_allegro/robots/dual_ur5e_allegro_inertia_changed.srdf" #for reference, srdf_path does not need to be set (may not actually even be used)
    fix_root_link = True
    disable_self_collisions = False # Set to True to disable self-collision
    arm_qpos = np.zeros(12)
    arm_qpos[::2] = left_arm_init_qpos
    arm_qpos[1::2] = right_arm_init_qpos

    # add a cube
    # object_urdf_path = "assets/urdf/objects/cube.urdf"
    # object_init_pose = sapien.Pose(p=[0.61, 0.17, 1.3], q=[0, -0.7071068, 0, 0.7071068])

    keyframes = dict(
        init=Keyframe(
            qpos=np.array(
                [
                    *arm_qpos,  # arm
                    *left_hand_init_qpos,  # left hand
                    *right_hand_init_qpos,  # right hand
                ]
            ),
            pose=sapien.Pose(p=[0, 0, 1.2]),
        )
    )

    left_arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    right_arm_joint_names = [joint + "_r" for joint in left_arm_joint_names]
    arm_joint_names = left_arm_joint_names + right_arm_joint_names

    left_hand_joint_names = ["joint_{}.0".format(i) for i in range(16)]
    right_hand_joint_names = ["joint_{}.0_r".format(i) for i in range(16)]
    hand_joint_names = left_hand_joint_names + right_hand_joint_names

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100
    arm_friction = 0.1

    hand_stiffness = 1e2
    hand_damping = 10
    hand_force_limit = 10
    hand_friction = 0.5

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
            friction=self.arm_friction,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
            friction=self.arm_friction,
        )
        hand_pd_joint_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            lower=None,
            upper=None,
            stiffness=self.hand_stiffness,
            damping=self.hand_damping,
            force_limit=self.hand_force_limit,
            normalize_action=True,
            friction=self.hand_friction,
        )

        hand_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.hand_stiffness,
            damping=self.hand_damping,
            force_limit=self.hand_force_limit,
            use_delta=True,
            friction=self.hand_friction,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, hand=hand_pd_joint_delta_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, hand=hand_pd_joint_pos),
        )
        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    @property
    def _sensor_configs(self):
        hand_view_l = sapien_utils.look_at(
            torch.tensor([0.1, 0, -0.1]),
            torch.tensor([0, 0, 1]),
            [0, 0, 1],
        )
        hand_view_r = sapien_utils.look_at(
            torch.tensor([0.1, 0, -0.1]),
            torch.tensor([0, 0, 1]),
            [0, 0, 1],
        )
        back_view = sapien_utils.look_at([0.0, 0.1, 1.8], [0, 0.5, 1.0], [0, 0, 1])

        return [
            CameraConfig(
                uid="hand_camera",
                pose=hand_view_l,
                width=64,
                height=64,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["wrist_base"],
            ),
            CameraConfig(
                uid="hand_camera_r",
                pose=hand_view_r,
                width=64,
                height=64,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["wrist_base_r"],
            ),
            CameraConfig(
                uid="back_camera",
                pose=back_view,
                width=64,
                height=64,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            ),
        ]
