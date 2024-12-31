import sapien
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from bimanual_allegro import Bimanual_Allegro
from mani_skill.utils.building.ground import build_ground
from mani_skill.envs.utils.randomization.pose import random_quaternions
from sapien.core import pysapien
from typing import Union, Any, Dict
import torch
import numpy as np

from mani_skill.utils.io_utils import load_json
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

@register_env("Bimanual_Allegro_Cube", max_episode_steps=200)
class Env(BaseEnv):
    SUPPORTED_ROBOTS = ["Bimanual_Allegro"]

    agent: Union[Bimanual_Allegro]

    def __init__(
        self, *args, robot_uids="Bimanual_Allegro", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.table_height = 1.1
        self.initialized = False
        self.max_reward = 5.0

        self.right_hand_link = []
        self.left_hand_link = []
        self.left_hand_tip_link = []
        self.right_hand_tip_link = []

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

        links = self.agent.robot.links
        for link in links:
            if ".0" in link.name and "r" in link.name and "tip" not in link.name:
                self.right_hand_link.append(link)
            elif ".0" in link.name and "r" not in link.name and "tip" not in link.name:
                self.left_hand_link.append(link)
            elif ".0" in link.name and "r" in link.name and "tip" in link.name:
                self.right_hand_tip_link.append(link)
            elif ".0" in link.name and "r" not in link.name and "tip" in link.name:
                self.left_hand_tip_link.append(link)
        self.initialized = True

    def _load_scene(self, options: dict):
        build_ground(self.scene)

        table_builder = self.scene.create_actor_builder()
        table_builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0.5, self.table_height]),
            half_size=[0.7, 0.5, 0.02],
            material=sapien.render.RenderMaterial(
                base_color=[0.1, 0.1, 0.1, 1],
            ),
        )
        table_builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0.4, self.table_height]),
            half_size=[0.7, 0.3, 0.02],
            material=sapien.physx.PhysxMaterial(0.5, 0.3, 0.6),
        )
        self.table = table_builder.build_static(name="table")

        self.cube = actors.build_colorful_cube(
            scene=self.scene,
            half_size=0.05,
            color=[1, 0, 0, 1],
            name="cube",
            body_type="dynamic",
            add_collision=True,
        )


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_agent(env_idx)
        self._initialize_actor(env_idx)

    def _initialize_actor(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            cube_xyz = torch.tensor([0.1, 0.5, 1.25]).repeat(b, 1)
            cube_xyz = torch.rand((b, 3)) * 0.02 + cube_xyz
            cube_xyz[:, 2] = 1.25

            orn = random_quaternions(b, device=self.device)

            cube_pose = Pose.create_from_pq(p=cube_xyz, q=orn)

            self.cube.set_pose(cube_pose)

    def _initialize_agent(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            dof = self.agent.robot.dof
            if isinstance(dof, torch.Tensor):
                dof = dof[0]

            init_qpos = torch.tensor(
                self.agent.keyframes["init"].qpos, dtype=torch.float32
            ).repeat(b, 1)
            init_qpos += torch.randn((b, dof)) * self.robot_init_qpos_noise
            self.agent.reset(init_qpos)
            self.agent.robot.set_pose(self.agent.keyframes["init"].pose)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            cube_pos=self.cube.pose.p,
            cube_q=self.cube.pose.q,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                cube_ppos=self.cube.pose.p,
                cube_q=self.cube.pose.q,
            )
        return obs

    def evaluate(self):
        if self.initialized:
            right_hand_link_z = torch.concat(
                [
                    link.pose.p[..., 2].unsqueeze(1)
                    for link in self.right_hand_link + self.right_hand_tip_link
                ],
                dim=1,
            )

            fail_collision_table = (right_hand_link_z < self.table_height + 0.04).any(
                dim=1
            )
            fail_cube_fall = self.cube.pose.p[:, 2] < 1.0
            fail = fail_collision_table | fail_cube_fall
        else:
            fail = torch.zeros_like(self.cube.pose.p[:, 0], dtype=torch.bool)
        
        # Create success condition
        success = self.cube.pose.p[:, 2] >= 1.3
        
        # Calculate reward directly here instead of calling compute_dense_reward
        reward = torch.zeros_like(self.cube.pose.p[:, 0], device=self.device)
        reward[success] = self.max_reward
        reward[fail] = -self.max_reward / 4
        
        state = {
            "success": success,
            "fail": fail,
            "episode": {"r": reward}
        }
        return state


    def compute_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        total_reward = torch.zeros(len(obs), device=self.device)

        cube_xyz = self.cube.pose.p

        lift_reward = cube_xyz[:, 2] - self.table_height - 0.07
        total_reward += lift_reward * 20

        right_hand_tip_link_pose = torch.concat(
            [link.pose.p.unsqueeze(1) for link in self.right_hand_tip_link], dim=1
        ).to(self.device)

        right_hand_cube_distance = torch.linalg.norm(
            right_hand_tip_link_pose - cube_xyz.unsqueeze(1), axis=-1
        )

        hand_close_reward = torch.clamp(
            0.05 / right_hand_cube_distance, min=0, max=1.0
        ).mean(dim=-1)

        right_hand_tip_center = right_hand_tip_link_pose.mean(dim=1)

        right_hand_tip_center_distance = torch.linalg.norm(
            right_hand_tip_center - cube_xyz, axis=-1
        )

        center_close_reward = torch.clamp(
            0.025 / right_hand_tip_center_distance, min=0, max=1.0
        )

        total_reward += hand_close_reward + center_close_reward

        total_reward = total_reward.clamp(-self.max_reward, self.max_reward)

        total_reward[info["success"]] = self.max_reward
        total_reward[info["fail"]] = -self.max_reward / 4
        print(
            "total: {:.2f}, lift: {:.2f}, center:{:.2f} hand_close: {:.2f}".format(
                total_reward[0].item(),
                lift_reward[0].item(),
                center_close_reward[0].item(),
                hand_close_reward[0].item(),
            ),
            end="\r",
        )

        return total_reward

    def compute_normalized_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        self.max_reward = 5.0
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        norm_dense_reward = dense_reward / (2 * self.max_reward) + 0.5
        return norm_dense_reward

    # Define camera configurations for rendering and capturing videos (training & evaluation)
    @property
    def _default_sensor_configs(self):
        # Set up a camera for observations during training
        pose = sapien_utils.look_at(eye=[0.5, 1.5, 2.0], 
                                    target=[0.0, 0.5, self.table_height])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        # Set up a high-definition camera for rendering and video recording
        pose = sapien_utils.look_at(eye=[0.5, 1.5, 2.0], 
                                    target=[0.0, 0.5, self.table_height])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )
