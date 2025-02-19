import numpy as np

from sim.mpc.group_nopred_mpc import GroupNoPredMPC
from sim.mpc.group import draw_all_social_spaces
from sim.mpc.group_shape_prediction import GroupShapePrediction

class GroupConvMPC(GroupNoPredMPC):
    # MPC class for Group-based representation with Convolutional prediction
    def __init__(self, args, logger, dataset_info, conv_model_path):
        # MPC parameters
        super(GroupConvMPC, self).__init__(args, logger, dataset_info)
        if self.future_steps != 8:
            self.logger.error('Conv model only supports 8 future steps')
            raise ValueError('Conv model only supports 8 future steps')
        self.path = conv_model_path
        self.gpu_id = args.gpu_id

        self.predictor = GroupShapePrediction(self.dataset_info, self.path, self.logger, self.gpu_id)
        return

    def get_state_and_predictions(self, obs):
        # Get predictions for MPC
        # Linearly predict the future positions
        curr_pos = obs['pedestrians_pos']
        
        self.boundary_const = obs['personal_size']
        num_ped = len(curr_pos)

        self.frame_predictions = []
        self.boundary_predictions = []

        if not num_ped == 0:
            if not self.laser:
                curr_pos = obs['pedestrians_pos']
                curr_vel = obs['pedestrians_vel']
                history_pos = obs['pedestrians_pos_history']
                history_vel = obs['pedestrians_vel_history']
                group_ids = obs['group_labels']
                labels = group_ids
                self.frame_predictions = self.predictor.predict(history_pos, history_vel, group_ids, self.boundary_const, self.offset)
            else:
                curr_pos = obs['laser_pos']
                curr_vel = obs['laser_vel']
                history_pos = obs['laser_pos_history']
                history_vel = obs['laser_vel_history']
                group_ids = obs['laser_group_labels_history']
                labels = obs['laser_group_labels']
                self.frame_predictions = self.predictor.laser_predict(history_pos, history_vel, group_ids, self.dt, self.boundary_const, self.offset)

            for frame in self.frame_predictions:
                self.boundary_predictions.append(self._frame_to_vertices(self.dataset_info, frame))

            if self.animate and (not self.paint_boundary):
                self.boundary_pts = draw_all_social_spaces(labels, curr_pos, curr_vel, self.boundary_const, self.offset)

        return