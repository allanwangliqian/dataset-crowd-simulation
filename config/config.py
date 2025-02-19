import argparse
import sys
import torch
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='group_simulator')

    # save directory
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="exps/results",
        help="Path to save experiment results"
    )

    # environment configuration
    parser.add_argument(
        "--dset-file",
        type=str,
        default="config/datasets.yaml",
        help="file on which datasets to load"
    )

    parser.add_argument(
        "--dset-path",
        type=str,
        default="sim",
        help="base directory of the datasets"
    )

    parser.add_argument(
        "--ghost-time",
        type=float,
        default=1.0,
        help="time for the pedestrian to be ghosted after spawn"
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="dt of the simulator"
    )

    parser.add_argument(
        "--robot-speed",
        type=float,
        default=1.75,
        help="maximum robot speed (avg human walking speed)"
    )

    parser.add_argument(
        "--turn-speed",
        type=float,
        default=np.pi/2,
        help="maximum robot turning speed"
    )

    parser.add_argument(
        "--differential",
        action='store_true',
        default=False,
        help="if robot is differential drive"
    )

    parser.add_argument(
        "--collision_radius",
        type=float,
        default=0.5,
        help="in navigation smaller than this means collision"
    )

    parser.add_argument(
        "--goal_radius",
        type=float,
        default=1.0,
        help="in navigation smaller than this means goal reached"
    )

    # experiment configuration
    parser.add_argument(
        "--group",
        action='store_true',
        default=False,
        help="if grouping is enabled"
    )

    parser.add_argument(
        "--laser",
        action='store_true',
        default=False,
        help="if laser scan simulation is enabled"
    )

    parser.add_argument(
        "--pred",
        action='store_true',
        default=False,
        help="if prediction is enabled"
    )

    parser.add_argument(
        "--history",
        action='store_true',
        default=True,
        help="if history is considered"
    )

    parser.add_argument(
        "--react",
        action='store_true',
        default=False,
        help="if ORCA pedestrians is enabled"
    )

    parser.add_argument(
        "--animate",
        action='store_true',
        default=False,
        help="if results will be saved into a video"
    )

    parser.add_argument(
        "--record",
        action='store_true',
        default=True,
        help="if all the trajectories will be recorded for evaluation"
    )

    parser.add_argument(
        "--edge",
        action='store_true',
        default=False,
        help="if edge based group is enabled"
    )

    parser.add_argument(
        "--paint-boundary",
        action='store_true',
        default=False,
        help="turn off to paint boundary externaly of the simulator"
    )

    parser.add_argument(
        "--edge-offset",
        type=float,
        default=1.0,
        help="length of offset for the back edge"
    )

    parser.add_argument(
        "--pred-method",
        type=str,
        default=None,
        help="which prediction method to use, specific to group or not"
    )

    parser.add_argument(
        "--history-steps",
        type=int,
        default=8,
        help="number of history time steps to consider for prediction"
    )

    parser.add_argument(
        "--future-steps",
        type=int,
        default=8,
        help="number of future time steps to predict"
    )

    parser.add_argument(
        "--ped-size",
        type=float,
        default=0.5,
        help="size of the pedestrian"
    )

    # Simulated lidar parameters
    # Default is SICK LMS511 2D Lidar
    parser.add_argument(
        "--laser-res",
        type=float,
        default=0.25 / 180 * np.pi,
        help="angle resolution of the simulated lidar"
    )

    parser.add_argument(
        "--laser-range",
        type=float,
        default=80.0,
        help="range of the simulated lidar"
    )

    parser.add_argument(
        "--laser-noise",
        type=float,
        default=0.05,
        help="positional noise of the lidar scan point"
    )

    # MPC configuration
    parser.add_argument(
        "--num-directions",
        type=int,
        default=12,
        help="number of general direction rollouts for MPC"
    )

    parser.add_argument(
        "--num-linear",
        type=int,
        default=10,
        help="number of linear velocity levels for MPC"
    )

    parser.add_argument(
        "--num-angular",
        type=int,
        default=10,
        help="number of angular velocity levels for MPC"
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="discount factor for cost estimation"
    )

    parser.add_argument(
        "--dist-weight",
        type=float,
        default=0.65,
        help="weight for the distance cost term"
    )

    parser.add_argument(
        "--goal-weight",
        type=float,
        default=0.35,
        help="weight for the goal cost term"
    )

    # device configuration
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU device ID')

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training/inference')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    args.fps = 1 / args.dt
    args.time_horizon = args.dt * args.future_steps
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

def check_args(args, logger):
    if args.cuda:
        logger.info("GPU enabled")
    else:
        logger.info("GPU disabled")
        logger.error("GPU is required for this simulator")
        raise Exception("GPU is required for this simulator")

    if args.pred:
        
        if args.group:
            if (not args.pred_method in ["group", "sgan", "linear", "edge"]):
                logger.error("Invalid prediction method name")
                raise Exception("Invalid prediction method name")
        else:
            if (not args.pred_method in ["sgan", "linear"]):
                logger.error("Invalid prediction method name")
                raise Exception("Invalid prediction method name")
            
        if (not args.pred_method == "linear") and (not args.history):
            logger.error("Trajectory prediction requires history")
            raise Exception("Trajectory prediction requires history")
        
        if (args.pred_method == "edge") and (not args.edge):
            logger.error("Edge prediction requires edge grouping")
            raise Exception("Edge prediction requires edge grouping")
        
        if (args.pred_method == "group") and (not args.group):
            logger.error("Group prediction requires grouping")
            raise Exception("Group prediction requires grouping")
        
        if (args.pred_method == "sgan") and (args.laser):
            logger.error("SGAN prediction does not support simulated lidar")
            raise Exception("SGAN prediction does not support simulated lidar")
        
        if (args.pred_method == "sgan") and not((args.future_steps == 8) or (args.future_steps == 12)):
            logger.error("SGAN prediction requires 8 or 12 future steps")
            raise Exception("SGAN prediction requires 8 or 12 future steps")
        
        if (args.edge) and (args.edge_offset <= 0):
            logger.error("Edge offset must be positive")
            raise Exception("Edge offset must be positive")

    return