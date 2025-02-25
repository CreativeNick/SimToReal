import sapien
from mani_skill import ASSET_DIR
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

from mani_skill.utils.structs import Actor

@register_env("Bimanual_Allegro_YCB", max_episode_steps=200)
class Env(BaseEnv):
    SUPPORTED_ROBOTS = ["Bimanual_Allegro"]

    agent: Union[Bimanual_Allegro]

    def __init__(
        self, *args,
        robot_uids="Bimanual_Allegro",
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        # # load all YCB model IDs before initializion
        # self.all_model_ids = np.array(
        #     list(
        #         load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json").keys()
        #     )
        # )

        all_possible_models = list(
            load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json").keys()
        )

        print(f"Total available YCB objects: {len(all_possible_models)}")

        self.all_model_ids = np.array(all_possible_models[2:3])
        print(f"Using {len(self.all_model_ids)} YCB objects:")
        for i, model_id in enumerate(self.all_model_ids):
            print(f"{i+1}. {model_id}")


        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.table_height = 1.1
        self.initialized = False

        self.right_hand_link = []
        self.left_hand_link = []
        self.left_hand_tip_link = []
        self.right_hand_tip_link = []

        self.max_reward = 5.0

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

    def reconfigure(self):
        """Called by the training loop to reconfigure environment"""
        if hasattr(self.scene, 'reconfigure'):
            self.scene.reconfigure()
        # After reconfiguration, re-initialize the episode
        self._initialize_episode(torch.arange(self.num_envs, device=self.device), {})

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
            pose=sapien.Pose(p=[0, 0.5, self.table_height]),
        	half_size=[0.7, 0.5, 0.02],
            material=sapien.physx.PhysxMaterial(0.5, 0.3, 0.6),
        )
        self.table = table_builder.build_static(name="table")

        # sample YCB objects for each parallel environment
        model_ids = self._batched_episode_rng.choice(self.all_model_ids)
        
        stored_actors = []  # Renamed from 'actors' to avoid conflict
        for i, model_id in enumerate(model_ids):
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")  # Using actors module
            builder.set_scene_idxs([i])
            stored_actors.append(builder.build(name=f"{model_id}-{i}"))

        # merge actors for efficient handling
        self.ycb_object = Actor.merge(stored_actors, name="ycb_object")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_agent(env_idx)
        self._initialize_actor(env_idx)

    def _initialize_actor(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            #print(f"Initializing {b} YCB objects")
            
            # Set initial position to middle of desired range
            # x: default position on table (0.0)
            # y: default position on table (0.5)
            # z: slightly above table surface
            ycb_xyz = torch.tensor([0.0, 0.5, self.table_height + 0.1]).repeat(b, 1)
            #print(f"Initial positions: {ycb_xyz}")
            
            # Generate random offsets for x and y coordinates
            x_rand = np.random.uniform(-0.7, 0.7)
            y_rand = np.random.uniform(-0.4, 0.25)

            # Adjust multipliers to match desired range
            # x: random range of -0.7 to 0.7 based on x_rand
            # y: random range of -0.4 to 0.25 based on y_rand
            # z: no randomness, keep it 0
            ycb_xyz = (torch.rand((b, 3)) * 0.5) * torch.tensor([x_rand, y_rand, 0.0]) + ycb_xyz
        
            # Set z to table height + 0.1 to avoid clipping through the table
            ycb_xyz[:, 2] = self.table_height + 0.1

            # Generate random orientations for the YCB object
            orn = random_quaternions(b, device=self.device)

            ycb_pose = Pose.create_from_pq(p=ycb_xyz, q=orn)
            self.ycb_object.set_pose(ycb_pose)

            #print(f"Final YCB poses: {self.ycb_object.pose.p}")

    def _initialize_agent(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            
            # create full qpos array (44 total for initialization)
            init_qpos = torch.tensor(
                self.agent.keyframes["init"].qpos, dtype=torch.float32
            ).repeat(b, 1)

            # get indices for right arm/hand joints
            right_indices = []
            for i, joint in enumerate(self.agent.robot.get_active_joints()):
                if "_r" in joint.name:  # only right side joints
                    right_indices.append(i)

            # only add noise to right arm indices
            init_qpos[:, right_indices] += (
                torch.randn((b, len(right_indices)), device=self.device) * 
                self.robot_init_qpos_noise
            )
            
            self.agent.reset(init_qpos)
            self.agent.robot.set_pose(self.agent.keyframes["init"].pose)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            ycb_pos=self.ycb_object.pose.p,
            ycb_q=self.ycb_object.pose.q,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                ycb_ppos=self.ycb_object.pose.p,
                ycb_q=self.ycb_object.pose.q,
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

            fail_collision_table = (right_hand_link_z < self.table_height + 0.02).any(
                dim=1
            )
            fail_ycb_fall = self.ycb_object.pose.p[:, 2] < self.table_height - 0.02
            fail = fail_collision_table | fail_ycb_fall
        else:
            fail = torch.zeros_like(self.ycb_object.pose.p[:, 0], dtype=torch.bool)
            
        # create success condition
        success = self.ycb_object.pose.p[:, 2] >= 1.25
        
        # calculate reward directly here instead of calling compute_dense_reward
        reward = torch.zeros_like(self.ycb_object.pose.p[:, 0], device=self.device)
        reward[success] = self.max_reward
        reward[fail] = -self.max_reward / 4
        
        state = {
            "success": success,
            "fail": fail,
            "episode": {"r": reward}
        }
        return state

    def quat_to_rot(self, q):
        # q is a tensor of shape (..., 4) in the format [w, x, y, z]
        w, x, y, z = q.unbind(-1)
        R00 = 1 - 2*(y**2 + z**2)
        R01 = 2*(x*y - z*w)
        R02 = 2*(x*z + y*w)
        R10 = 2*(x*y + z*w)
        R11 = 1 - 2*(x**2 + z**2)
        R12 = 2*(y*z - x*w)
        R20 = 2*(x*z - y*w)
        R21 = 2*(y*z + x*w)
        R22 = 1 - 2*(x**2 + y**2)
        R = torch.stack([torch.stack([R00, R01, R02], dim=-1),
                        torch.stack([R10, R11, R12], dim=-1),
                        torch.stack([R20, R21, R22], dim=-1)], dim=-2)
        return R

    def compute_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        batch_size = self.ycb_object.pose.p.shape[0]
        #print(f"Batch size: {batch_size}")
        total_reward = torch.zeros(batch_size, device=self.device)

        ycb_xyz = self.ycb_object.pose.p
        #print(f"Shape of ycb_xyz: {ycb_xyz.shape}")

        # step 1: approach reward
        # compute the center of the right-hand tip links
        right_hand_tip_positions = torch.cat(
            [link.pose.p.unsqueeze(1) for link in self.right_hand_tip_link], dim=1
        )  # shape: (batch_size, num_tips, 3)
        hand_center = right_hand_tip_positions.mean(dim=1)  # (batch_size, 3)
        approach_distance = torch.linalg.norm(hand_center - ycb_xyz, dim=-1)
        approach_reward = torch.exp(-2.0 * approach_distance)
        total_reward += approach_reward * 2.0

        # step 2: grasp reward
        # reward based on fingertips being close to the object
        finger_tip_distances = torch.linalg.norm(
            right_hand_tip_positions - ycb_xyz.unsqueeze(1), dim=-1
        )  # (batch_size, num_tips)
        grasp_proximity_reward = torch.exp(-5.0 * finger_tip_distances).mean(dim=1)
        # reward for a closed hand
        finger_positions = torch.stack([link.pose.p for link in self.right_hand_link], dim=1)
        finger_spread = torch.std(finger_positions, dim=1).mean(dim=-1)

        desired_spread = 0.05 # in meters, *100 for cm
        spread_error = torch.abs(finger_spread - desired_spread)

        optimal_spread_reward = torch.exp(-10.0 * spread_error)
        grasp_reward = grasp_proximity_reward * optimal_spread_reward
        total_reward += grasp_reward * 1.0

        # step 3: lift reward
        lift_baseline = self.table_height + 0.1
        lift_reward = torch.clamp(ycb_xyz[:, 2] - lift_baseline, min=0.0)
        total_reward += lift_reward * 15.0
        high_lift_threshold = self.table_height + 0.2
        high_lift_reward = torch.where(
            ycb_xyz[:, 2] > high_lift_threshold,
            (ycb_xyz[:, 2] - high_lift_threshold) * 25.0,
            torch.zeros_like(ycb_xyz[:, 2])
        )
        total_reward += high_lift_reward

        # step 4: orientation reward
        # self.right_hand_link[0] corresponds to the palm
        # reward the agent when the palm is flat and facing downwards ([0, 0, -1]) towards the object
        
        palm_q = self.right_hand_link[0].pose.q  # get the quaternion from pose
        #print("PALM Q: ", palm_q)
        palm_R = self.quat_to_rot(palm_q) # convert to rotation matrix

        local_palm_normal = torch.tensor([0, 0, -1], device=self.device, dtype=torch.float32)
        local_palm_normal_expanded = local_palm_normal.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # shape: (batch_size, 3, 1)
        palm_normal = torch.matmul(palm_R, local_palm_normal_expanded).squeeze(-1)  # shape: (batch_size, 3)
        desired_direction = torch.tensor([0, 0, -1], device=self.device, dtype=torch.float32)
        orientation_dot = torch.sum(palm_normal * desired_direction, dim=1)  # shape: (batch_size,)
        orientation_reward = torch.clamp(orientation_dot, 0.0, 1.0)
        total_reward += orientation_reward * 1.0

        # left-arm movement penalty
        if hasattr(self, "initial_left_hand_positions"):
            left_hand_positions = torch.stack(
                [link.pose.p for link in self.left_hand_link], dim=1
            )
            left_movement = torch.linalg.norm(
                left_hand_positions - self.initial_left_hand_positions, dim=-1
            ).mean(dim=-1)
            total_reward += (-left_movement * 100.0)

        total_reward = total_reward.clamp(-self.max_reward, self.max_reward)
        total_reward[info["success"]] = self.max_reward
        total_reward[info["fail"]] = -self.max_reward / 4

        return total_reward
    
    # def compute_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
    #     batch_size = self.ycb_object.pose.p.shape[0]
    #     #print(f"Batch size: {batch_size}")
    #     total_reward = torch.zeros(batch_size, device=self.device)

    #     ycb_xyz = self.ycb_object.pose.p
    #     #print(f"Shape of ycb_xyz: {ycb_xyz.shape}")

    #     lift_reward = ycb_xyz[:, 2] - self.table_height - 0.07
    #     #print(f"Shape of lift_reward: {lift_reward.shape}")
    #     #print(f"Shape of total_reward: {total_reward.shape}")
    #     total_reward += lift_reward * 15

    #     # penalty for left arm movement
    #     left_hand_positions = torch.stack([link.pose.p for link in self.left_hand_link], dim=1)
    #     left_hand_movement = torch.linalg.norm(left_hand_positions - left_hand_positions.detach(), dim=-1).mean(dim=-1)
    #     left_arm_penalty = -left_hand_movement * 50.0
    #     total_reward += left_arm_penalty

    #     # get finger tip pose
    #     right_hand_tip_link_pose = torch.concat(
    #         [link.pose.p.unsqueeze(1) for link in self.right_hand_tip_link], dim=1
    #     ).to(self.device)

    #     # finger spread reward to encourage finger movement
    #     finger_positions = torch.stack([link.pose.p for link in self.right_hand_link], dim=1)
    #     finger_spread = torch.std(finger_positions, dim=1).mean(dim=-1)
    #     finger_spread_reward = torch.clamp(finger_spread * 5.0, 0, 1.0)
    #     total_reward += finger_spread_reward

    #     # calculate distance between finger tip and ycb object
    #     right_hand_ycb_distance = torch.linalg.norm(
    #         right_hand_tip_link_pose - ycb_xyz.unsqueeze(1), axis=-1
    #     )

    #     # calculate reward based on distance
    #     #hand_close_reward = torch.clamp(0.05 / right_hand_ycb_distance, min=0, max=1.0).mean(dim=-1)
    #     hand_close_reward = torch.exp(-2.0 * right_hand_ycb_distance).mean(dim=-1)
    #     total_reward += hand_close_reward * 2.0

    #     right_hand_tip_center = right_hand_tip_link_pose.mean(dim=1)
    #     right_hand_tip_center_distance = torch.linalg.norm(
    #         right_hand_tip_center - ycb_xyz, axis=-1
    #     )

    #     #center_close_reward = torch.clamp(0.025 / right_hand_tip_center_distance, min=0, max=1.0)
    #     center_close_reward = torch.exp(-2.0 * right_hand_tip_center_distance)
    #     total_reward += hand_close_reward + center_close_reward

    #     # calculate height-based reward
    #     height_diff = ycb_xyz[:, 2] - self.table_height
    #     height_reward = torch.where(height_diff > 0, height_diff * 10.0, torch.zeros_like(height_diff))
    #     total_reward += height_reward

    #     total_reward = total_reward.clamp(-self.max_reward, self.max_reward)

    #     total_reward[info["success"]] = self.max_reward
    #     total_reward[info["fail"]] = -self.max_reward / 4
    #     print(
    #         "total: {:.2f}, lift: {:.2f}, center:{:.2f} hand_close: {:.2f}".format(
    #             total_reward[0].item(),
    #             lift_reward[0].item(),
    #             center_close_reward[0].item(),
    #             hand_close_reward[0].item(),
    #         ),
    #         end="\r",
    #     )

    #     return total_reward

    def compute_normalized_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        self.max_reward = 5.0
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        norm_dense_reward = dense_reward / (2 * self.max_reward) + 0.5
        return norm_dense_reward
    
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, 1.5, 2.0], 
                                    target=[0.0, 0.5, self.table_height])
        print(f"Camera pose: {pose}")
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
        pose = sapien_utils.look_at(eye=[0.5, 1.5, 2.0], 
                                    target=[0.0, 0.5, self.table_height])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )
