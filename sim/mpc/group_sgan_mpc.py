import numpy as np

from sim.mpc.group_nopred_mpc import GroupNoPredMPC
from sgan.scripts.inference import SGANInference

class GroupSGANMPC(GroupNoPredMPC):
    # MPC class for Group-based representation with SGAN prediction
    def __init__(self, args, logger, sgan_model_path):
        # MPC parameters
        super(GroupSGANMPC, self).__init__(args, logger)
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
        group_ids = obs['group_labels']
        self.dataset_info = obs['dataset_info']
        self.boundary_const = obs['personal_size']
        num_ped = len(curr_pos)

        self.frame_predictions = []
        self.boundary_predictions = []

        if not num_ped == 0:
            pos_predictions = self.sgan.evaluate(history_pos)
            vel_predictions = np.zeros_like(pos_predictions)
            vel_predictions[:, 1:, :] = (pos_predictions[:, 1:, :] - pos_predictions[:, :-1, :]) / self.dt
            vel_predictions[:, 0, :] = (pos_predictions[:, 1, :] - curr_pos) / self.dt

            for i in range(self.future_steps):
                frame = self._get_frame(self.dataset_info, pos_predictions[:, i, :], vel_predictions[:, i, :], group_ids, self.boundary_const)
                self.frame_predictions.append(frame)
                self.boundary_predictions.append(self._frame_to_vertices(self.dataset_info, frame))

        return