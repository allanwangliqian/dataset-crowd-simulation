import numpy as np
from sim.mpc.ped_nopred_mpc import PedNoPredMPC
from sgan.scripts.inference import SGANInference

class PedSGANMPC(PedNoPredMPC):
    # MPC class for Pedestrian-based representation with SGAN prediction
    def __init__(self, args, logger, sgan_model_path):
        # MPC parameters
        super(PedSGANMPC, self).__init__(args, logger)
        if args.laser:
            self.logger.error('SGAN model does not support laser input')
            raise ValueError('SGAN model does not support laser input')
        if args.history_steps != 8:
            self.logger.error('SGAN model only supports 8 history steps')
            raise ValueError('SGAN model only supports 8 history steps')
        future_intend = sgan_model_path.split('_')[-2]
        if int(future_intend) != self.future_steps:
            self.logger.error('SGAN model does not support the future steps')
            raise ValueError('SGAN model does not support the future steps')
        self.sgan = SGANInference(sgan_model_path, args.gpu_id)
        return

    def get_state_and_predictions(self, obs):
        # Get predictions for MPC
        # Linearly predict the future positions
        curr_pos = obs['pedestrians_pos']
        history_pos = obs['pedestrians_pos_history']
        self.boundary_const = obs['personal_size']
        num_ped = len(curr_pos)
        if num_ped == 0:
            self.pos_predictions = []
            self.vel_predictions = []
        else:
            self.pos_predictions = self.sgan.evaluate(history_pos)
            self.vel_predictions = np.zeros_like(self.pos_predictions)
            self.vel_predictions[:, 1:, :] = (self.pos_predictions[:, 1:, :] - self.pos_predictions[:, :-1, :]) / self.dt
            self.vel_predictions[:, 0, :] = (self.pos_predictions[:, 0, :] - curr_pos) / self.dt
        return