import os, sys
import logging
import yaml
import numpy as np

from config import get_args, check_args
from sim.simulator import Simulator
from sim.mpc.ped_nopred_mpc import PedNoPredMPC
from sim.mpc.ped_linear_mpc import PedLinearMPC
from sim.mpc.ped_sgan_mpc import PedSGANMPC

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
    args.group = False
    args.react = False
    args.laser = False
    args.record = True
    args.animate = True
    args.history = True
    args.differential = True
    
    sim = Simulator(args, 'data/eth_0.json', logger)
    # agent = PedNoPredMPC(args, logger)
    # agent = PedLinearMPC(args, logger)
    agent = PedSGANMPC(args, logger, 'sgan/models/sgan-models/eth_8_model.pt')
    obs = sim.reset(100)
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = sim.step(action)
    sim.evaluate(output=True)
