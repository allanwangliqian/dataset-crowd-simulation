import numpy as np
from sim.mpc.ped_nopred_mpc import PedNoPredMPC
from sim.mpc.group import draw_all_social_spaces

class PedLinearMPC(PedNoPredMPC):
    # MPC class for Pedestrian-based representation with linear prediction
    def __init__(self, args, logger):
        # MPC parameters
        super(PedLinearMPC, self).__init__(args, logger)
        return

    def get_state_and_predictions(self, obs):
        # Get predictions for MPC
        # Linearly predict the future positions
        if self.laser:
            curr_pos = obs['laser_pos']
            curr_vel = obs['laser_vel']
        else:
            curr_pos = obs['pedestrians_pos']
            curr_vel = obs['pedestrians_vel']
        self.boundary_const = obs['personal_size']

        if len(curr_pos) == 0:
            self.pos_predictions = []
            self.vel_predictions = []
        else:
            self.pos_predictions = np.zeros((len(curr_pos), self.future_steps, 2))
            self.vel_predictions = np.zeros((len(curr_pos), self.future_steps, 2))
            for i in range(self.future_steps):
                self.pos_predictions[:, i, :] = curr_pos + curr_vel * (i + 1) * self.dt
                self.vel_predictions[:, i, :] = curr_vel

            if self.animate and (not self.paint_boundary):
                group_ids = list(range(len(curr_pos)))
                self.boundary_pts = draw_all_social_spaces(group_ids, curr_pos, curr_vel, self.boundary_const, self.offset)
        return