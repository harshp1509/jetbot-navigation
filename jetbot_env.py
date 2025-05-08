import math
import torch
import os
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import isaaclab.sim as sim_utils

import isaaclab_tasks.manager_based.navigation.mdp as mdp
# from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg
# from isaaclab_tasks.manager_based.navigation.mdp import JetBotEnvCfg 
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.scene import InteractiveSceneCfg
# LOW_LEVEL_ENV_CFG = JetBotEnvCfg()

# LOW_LEVEL_ENV_CFG = AnymalCFlatEnvCfg()

JETBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=os.environ['HOME']+"/isaac_sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/jetbot/jetbot.usd",
        usd_path="/media/airangers/AiRangersData2/Harsh/jetbot.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=2.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.0, 0.0, 0.0), 
        joint_pos={"right_wheel_joint": 0.0, "left_wheel_joint": 0.0},
    ),
    actuators={
        "wheel_motors": ImplicitActuatorCfg(
            joint_names_expr=["right_wheel_joint", "left_wheel_joint"],
            effort_limit=20.0,
            velocity_limit=20.0,
            stiffness=800.0,
            damping=80.0,
        ),
    },
)

@configclass
class JetSceneCfg(InteractiveSceneCfg):
    """Configuration for a Jet collision avoidance scene using LiDAR."""

    # Jet entity
    robot: ArticulationCfg = JETBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Attach LiDAR to the jet
    # lidar: RayCasterCfg = LIDAR_CFG
    
    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )





@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    jet_controls = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["right_wheel_joint", "left_wheel_joint"],
        scale=10.0
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # linear_velocity = ObsTerm(func=mdp.base_lin_vel)  # Uses JetBot’s odometry
        # angular_velocity = ObsTerm(func=mdp.base_ang_vel)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # orientation=ObsTerm(func=mdp.root_quat_w)
        position=ObsTerm(func=mdp.root_pos_w)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        
        # camera_image = ObsTerm(func=mdp.camera_rgb, params={"sensor_name": "front_camera"})  # If using vision

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # position_tracking = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=1.0,  # Increase weight for better tracking
    #     params={"std": 1.5, "command_name": "pose_command"},
    # )
    # collision_penalty = RewTerm(
    #     func=mdp.collision_penalty_tanh, 
    #     weight=-5.0,  # Penalize collisions
    #     params={"sensor_name":"Lidar"}
    # )
    
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(20.0, 20.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-1.0, 1.0), pos_y=(-1.0, 1.0), heading=(-math.pi, math.pi)),
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # pass
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    # )

    # time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # collision = DoneTerm(
    #     func=mdp.collision_penalty_tanh, params={"sensor_name": "collision_sensor"}
    # )


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # environment settings
    scene:  JetSceneCfg = JetSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""

        self.decimation = 2
        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.render_interval = self.decimation
