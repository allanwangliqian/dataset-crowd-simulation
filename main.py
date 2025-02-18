import os, sys
import logging
import yaml
import pickle
from time import time

from config.config import get_args, check_args
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

    results = []
    sim = Simulator(args, 'data/all.json', logger)
    obs = sim.reset()
    while not (obs is None):
        case_info = sim.get_case_info()

        if (args.pred_method == 'sgan') or (args.pred_method == 'edge'):
            if (case_info['env_name'] == 'eth') and (case_info['env_flag'] == 0):
                sgan_model_path = "sgan/models/sgan-models/eth_" + str(args.future_steps) + "_model.pt"
            elif (case_info['env_name'] == 'eth') and (case_info['env_flag'] == 1):
                sgan_model_path = "sgan/models/sgan-models/hotel_" + str(args.future_steps) + "_model.pt"
            elif (case_info['env_name'] == 'ucy') and (case_info['env_flag'] == 0):
                sgan_model_path = "sgan/models/sgan-models/zara1_" + str(args.future_steps) + "_model.pt"
            elif (case_info['env_name'] == 'ucy') and (case_info['env_flag'] == 1):
                sgan_model_path = "sgan/models/sgan-models/zara2_" + str(args.future_steps) + "_model.pt"
            elif (case_info['env_name'] == 'ucy') and (case_info['env_flag'] == 2):
                sgan_model_path = "sgan/models/sgan-models/univ_" + str(args.future_steps) + "_model.pt"
            else:
                logger.error('Dataset is not supported by SGAN')
                raise ValueError('Dataset is not supported by SGAN')
            
        # set up the agent
        if args.group:
            raise NotImplementedError('Group simulation is not implemented yet')
        else:
            if not args.pred:
                agent = PedNoPredMPC(args, logger)
            else:
                if args.pred_method == 'linear':
                    agent = PedLinearMPC(args, logger)
                elif args.pred_method == 'sgan':
                    agent = PedSGANMPC(args, logger, sgan_model_path)
                else:
                    logger.error('Prediction method is not supported')
                    raise ValueError('Prediction method is not supported')
                
        done = False
        start_time = time()
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = sim.step(action)
        end_time = time()
        logger.info('Time spent: {}'.format(end_time - start_time))
        logger.info('Result: {}'.format(info))

        rst = sim.evaluate(output=False)
        history = sim.get_trial_history()
        state_time, eval_time = agent.get_processing_time()
        rst['state_time'] = state_time
        rst['eval_time'] = eval_time

        result_info = {'history': history, 'result': rst}
        results.append(result_info)

        with open(os.path.join(args.output_dir, "results.pickle"), "wb") as fp:
            pickle.dump(results, fp)

        obs = sim.reset()
