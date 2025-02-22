import os, sys
import logging
import yaml
import numpy as np
from time import time

sys.path.append('crowdattn')

from config.config import get_args, check_args
from sim.simulator import Simulator
from sim.mpc.ped_nopred_mpc import PedNoPredMPC
from sim.mpc.ped_linear_mpc import PedLinearMPC
from sim.mpc.ped_sgan_mpc import PedSGANMPC
from sim.mpc.group_nopred_mpc import GroupNoPredMPC
from sim.mpc.group_linear_mpc import GroupLinearMPC
from sim.mpc.group_sgan_mpc import GroupSGANMPC
from sim.mpc.group_conv_mpc import GroupConvMPC
from sim.mpc.group_edge_mpc import GroupEdgeMPC
from sim.crowd_attn_rl import CrowdAttnRL

if __name__ == "__main__":

    # configue and logs
    args = get_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    log_fname = os.path.join(args.output_dir, 'experiment.log')
    file_handler = logging.FileHandler(log_fname, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
						format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)

    check_args(args, logger)

    # which datasets to preload
    yaml_stream = open(args.dset_file, "r")
    yaml_dict = yaml.safe_load(yaml_stream)
    dsets = yaml_dict["datasets"]
    flags = yaml_dict["flags"]
    if not len(dsets) == len(flags):
        logger.error("datasets file - number of datasets and flags are not equal!")
        raise Exception("datasets file - number of datasets and flags are not equal!")
    
    envs_arg = []
    for i in range(len(dsets)):
        dset = dsets[i]
        flag = flags[i]
        envs_arg.append((dset, flag))
    args.envs = envs_arg

    # DANGER!!! temporarily configure args
    args.rl = True
    if not args.rl:
        args.group = True
        args.react = False
        args.laser = True
        args.edge = True
        args.record = True
        args.animate = True
        args.paint_boundary = False
        args.history = True
        args.future_steps = 8
        args.differential = False
    else:
        args.group = False
        args.react = False
        args.laser = False
        args.edge = False
        args.record = True
        args.animate = True
        args.paint_boundary = True
        args.history = True
        args.future_steps = 5
        args.differential = False
    
    sim = Simulator(args, 'data/ucy_2.json', logger)
    obs = sim.reset(189)
    dataset_info = obs['dataset_info']
    # agent = PedNoPredMPC(args, logger)
    # agent = PedLinearMPC(args, logger)
    # agent = PedSGANMPC(args, logger, 'sgan/models/sgan-models/eth_8_model.pt')
    # agent = GroupNoPredMPC(args, logger, dataset_info)
    # agent = GroupLinearMPC(args, logger, dataset_info)
    # agent = GroupSGANMPC(args, logger, dataset_info, 'sgan/models/sgan-models/eth_8_model.pt')
    # agent = GroupConvMPC(args, logger, dataset_info, 'checkpoints/model_conv_0.pth')
    # agent = GroupEdgeMPC(args, logger, dataset_info, 'sgan/models/sgan-models/zara2_8_model.pt')
    agent = CrowdAttnRL(args, logger, 'sgan/models/sgan-models/univ_8_model.pt', './crowdattn/trained_models/GST_predictor_rand')

    done = False
    start_time = time()
    while not done:
        action = agent.act(obs, done)
        obs, reward, done, info = sim.step(action)
        if args.animate and not args.paint_boundary:
            frame = sim.get_latest_render_frame()
            frame = agent.add_boundaries(frame)
            sim.update_latest_render_frame(frame)
    end_time = time()
    logger.info('Time spent: {}'.format(end_time - start_time))
    rst = sim.evaluate(output=False)
    history = sim.get_trial_history()
    state_time, eval_time = agent.get_processing_time()
    logger.info('State time: {}, Eval time: {}'.format(state_time, eval_time))
