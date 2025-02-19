import numpy as np

from sim.mpc.group_nopred_mpc import GroupNoPredMPC
from sim.mpc.group import draw_all_social_spaces
from sim.mpc.group_shape_prediction import GroupShapePrediction

class GroupLinearMPC(GroupNoPredMPC):
    # MPC class for Group-based representation with central-based linear prediction
    def __init__(self, args, logger):
        # MPC parameters
        super(GroupLinearMPC, self).__init__(args, logger)
        return

    def get_state_and_predictions(self, obs):
        # Get predictions for MPC
        # Linearly predict the future positions
        if self.laser:
            curr_pos = obs['laser_pos']
            curr_vel = obs['laser_vel']
            group_ids = obs['laser_group_labels']
        else:
            curr_pos = obs['pedestrians_pos']
            curr_vel = obs['pedestrians_vel']
            group_ids = obs['group_labels']

        self.dataset_info = obs['dataset_info']
        self.boundary_const = obs['personal_size']
        num_ped = len(curr_pos)

        predictor = GroupShapePrediction(self.dataset_info, None, self.logger)

        self.frame_predictions = []
        self.boundary_predictions = []

        if not num_ped == 0:
            self.frame_predictions = predictor.linear_predict(curr_pos, curr_vel, group_ids, self.future_steps, self.dt, self.boundary_const, self.offset)

            for frame in self.frame_predictions:
                self.boundary_predictions.append(self._frame_to_vertices(self.dataset_info, frame))

            if self.animate and (not self.paint_boundary):
                self.boundary_pts = draw_all_social_spaces(group_ids, curr_pos, curr_vel, self.boundary_const, self.offset)

        return