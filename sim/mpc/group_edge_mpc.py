import numpy as np

from sim.mpc.base_mpc import BaseMPC
from sim.mpc.group import draw_social_shapes
from sgan.scripts.inference import SGANInference

class GroupEdgeMPC(BaseMPC):
    # MPC class for Edge-based representation with node-based trajectory prediction
    def __init__(self, args, logger, sgan_model_path):
        # MPC parameters
        super(GroupEdgeMPC, self).__init__(args, logger)
        if self.laser:
            self.offset = args.ped_size
        else:
            self.offset = 0
        self.edge_offset = args.edge_offset

        self.dataset_info = None
        self.edge_predictions = None

        if args.history_steps != 8:
            self.logger.error('SGAN model only supports 8 history steps')
            raise ValueError('SGAN model only supports 8 history steps')
        future_intend = sgan_model_path.split('_')[-2]
        if int(future_intend) != self.future_steps:
            self.logger.error('SGAN model does not support the future steps')
            raise ValueError('SGAN model does not support the future steps')
        
        self.sgan = SGANInference(sgan_model_path, args.gpu_id)
        return
    
    def _find_least_dist(self, config, points):
        # Find the least distance between config and the points
        # Inputs:
        # config: the configuration
        # points: the points, dimension is Nx2
        if len(points) == 0:
            return 1e+9, None
        diff = points - config
        dist = np.linalg.norm(diff, axis=1)
        return np.min(dist), np.argmin(dist)
        
    
    def _find_left_right_edge(self, points, target_pt):
        # Find the left and right edge points with respect to the target point

        angles = np.arctan2(points[:, 1] - target_pt[1], points[:, 0] - target_pt[0])
        left_idx = np.argmax(angles)
        right_idx = np.argmin(angles)
        left_ang = angles[left_idx]
        right_ang = angles[right_idx]

        # correction for groups that crosses pi
        if (left_ang - right_ang) > np.pi:
            angles = np.where(angles < 0, angles + 2 * np.pi, angles)
            left_idx = np.argmax(angles)
            right_idx = np.argmin(angles)

        return left_idx, right_idx


    def id_edge_pts(self, robo_pos, ped_pos_series, ped_vel_series, group_ids):
        # Get the 3 edge points needed to generate edges to be predicted
        # Inputs:
        # robo_pos: the robot position
        # ped_pos_series: the pedestrian position series NxTx2
        # ped_vel_series: the pedestrian velocity series NxTx2
        # group_ids: the group ids of the current pedestrians N
        #
        # Outputs:
        # edge_pos_series: the edge position series 3NxTx2
        # edge_vel_series: the edge velocity series 3NxTx2
        # The first of the 3N points is the left point, 
        # the second is the closest point to the robot, 
        # and the third is the right point

        robo_pos = np.array(robo_pos)
        time_steps = np.shape(ped_pos_series)[1]
        edge_pos_series = []
        edge_vel_series = []

        all_labels = np.unique(group_ids)
        for curr_label in all_labels:
            group_idxes = np.where(group_ids == curr_label)[0]
            edge_group_pos_l = [] # right is the first edge to hit counter clockwise
            edge_group_pos_c = []
            edge_group_pos_r = []
            edge_group_vel_l = []
            edge_group_vel_c = []
            edge_group_vel_r = []
            for t in range(time_steps):
                # identify three pts per group
                ang_positions = ped_pos_series[group_idxes, t, :]
                dists = np.linalg.norm(ang_positions - robo_pos, axis=1)
                close_idx = group_idxes[np.argmin(dists)]

                l_idx, r_idx = self._find_left_right_edge(np.array(ang_positions), robo_pos)
                left_idx = group_idxes[l_idx]
                right_idx = group_idxes[r_idx]
                edge_group_pos_l.append(ped_pos_series[left_idx][t])
                edge_group_pos_c.append(ped_pos_series[close_idx][t])
                edge_group_pos_r.append(ped_pos_series[right_idx][t])
                edge_group_vel_l.append(ped_vel_series[left_idx][t])
                edge_group_vel_c.append(ped_vel_series[close_idx][t])
                edge_group_vel_r.append(ped_vel_series[right_idx][t])
            edge_pos_series.append(edge_group_pos_l)
            edge_pos_series.append(edge_group_pos_c)
            edge_pos_series.append(edge_group_pos_r)
            edge_vel_series.append(edge_group_vel_l)
            edge_vel_series.append(edge_group_vel_c)
            edge_vel_series.append(edge_group_vel_r)

        return np.array(edge_pos_series), np.array(edge_vel_series)

    def id_edge_pts_laser(self, robo_pos, positions, velocities, group_ids, dt):
        # Get the 3 edge points needed to generate edges to be predicted from laser data
        # Inputs:
        # robo_pos: the robot position
        # positions: the laser point position series TxNx2
        # velocities: the laser point velocity series TxNx2
        # group_ids: the group ids of the laser points TxN
        # dt: the time interval between each frame
        #
        # Outputs:
        # edge_pos_series: the edge position series 3NxTx2
        # edge_vel_series: the edge velocity series 3NxTx2
        # The first of the 3N points is the left point, 
        # the second is the closest point to the robot, 
        # and the third is the right point

        # Nearest geo-center way of building history
        robo_pos = np.array(robo_pos)
        edge_pos_series = []
        edge_vel_series = []

        time_steps = len(positions)
        group_pos_series = []
        group_vel_series = []
        group_centers = []
        group_vel_centers = []
        # Get group scan pts, vels, centers & center_vels for each frame
        for i in range(time_steps):
            pos = positions[i]
            vel = velocities[i]
            labels = group_ids[i]
            all_labels = np.unique(labels)
            
            all_group_pos = []
            all_group_vel = []
            centers = []
            vel_centers = []
            for j, curr_label in enumerate(all_labels):
                idxes = np.where(labels == curr_label)[0]
                group_positions = pos[idxes]
                group_velocities = vel[idxes]
                center_x = np.mean(group_positions[:, 0])
                center_y = np.mean(group_positions[:, 1])
                center_vx = np.mean(group_velocities[:, 0])
                center_vy = np.mean(group_velocities[:, 1])

                all_group_pos.append(group_positions)
                all_group_vel.append(group_velocities)
                centers.append(np.array([center_x, center_y]))
                vel_centers.append(np.array([center_vx, center_vy]))
            group_pos_series.append(all_group_pos)
            group_vel_series.append(all_group_vel)
            group_centers.append(centers)
            group_vel_centers.append(vel_centers)

        temp_threshold = 2.5 * dt #m/s x s
        num_curr_groups = len(group_pos_series[-1])
        for i in range(num_curr_groups):
            position_seq = [group_pos_series[-1][i]]
            velocity_seq = [group_vel_series[-1][i]]
            config = group_centers[-1][i]
            break_idx = None
            save_idx = i
            # search nearest centers for each prev frame
            for j in range(time_steps-2, -1, -1):
                points = group_centers[j]
                min_dist, min_idx = self._find_least_dist(config, points)
                if min_dist > temp_threshold:
                    break_idx = j
                    break
                else:
                    position_seq.append(group_pos_series[j][min_idx])
                    velocity_seq.append(group_vel_series[j][min_idx])
                    config = group_centers[j][min_idx]
                    save_idx = min_idx
            # if discrepancy, linear back-prop
            if not (break_idx == None):
                position_last = group_pos_series[break_idx + 1][save_idx]
                velocity_last = group_vel_series[break_idx + 1][save_idx]
                vel = group_vel_centers[break_idx + 1][save_idx]
                for j in range(break_idx, -1, -1):
                    position_last = list(np.array(position_last) - vel / 10)
                    position_seq.append(position_last)
                    velocity_seq.append(velocity_last)
            
            edge_group_pos_l = [] # right is the first edge to hit counter clockwise
            edge_group_pos_c = []
            edge_group_pos_r = []
            edge_group_vel_l = []
            edge_group_vel_c = []
            edge_group_vel_r = []
            for t in range(time_steps-1, -1, -1):
                # identify three pts per group
                ang_positions = position_seq[t]
                dists = np.linalg.norm(ang_positions - robo_pos, axis=1)
                close_idx = np.argmin(dists)
                
                left_idx, right_idx = self._find_left_right_edge(np.array(ang_positions), robo_pos)
                edge_group_pos_l.append(position_seq[t][left_idx])
                edge_group_pos_c.append(position_seq[t][close_idx])
                edge_group_pos_r.append(position_seq[t][right_idx])
                edge_group_vel_l.append(velocity_seq[t][left_idx])
                edge_group_vel_c.append(velocity_seq[t][close_idx])
                edge_group_vel_r.append(velocity_seq[t][right_idx])
            edge_pos_series.append(edge_group_pos_l)
            edge_pos_series.append(edge_group_pos_c)
            edge_pos_series.append(edge_group_pos_r)
            edge_vel_series.append(edge_group_vel_l)
            edge_vel_series.append(edge_group_vel_c)
            edge_vel_series.append(edge_group_vel_r)

        return np.array(edge_pos_series), np.array(edge_vel_series)

    def _construct_offset(self, left_pt, right_pt, center_pt, offset=1.0):
        # Construct the offset points for the left and right points
        # Inputs:
        # left_pt: the left point coordinate
        # right_pt: the right point coordinate
        # center_pt: the center (robot) point coordinate
        # offset: the offset distance of the back edge

        ang1 = np.arctan2(left_pt[1] - right_pt[1], left_pt[0] - right_pt[0]) + np.pi/2
        ang2 = ang1 - np.pi
        offset1 = np.array([offset * np.cos(ang1), offset * np.sin(ang1)])
        offset2 = np.array([offset * np.cos(ang2), offset * np.sin(ang2)])
        left_pt_cd1 = left_pt + offset1
        left_pt_cd2 = left_pt + offset2
        right_pt_cd1 = right_pt + offset1
        right_pt_cd2 = right_pt + offset2

        dist1 = np.linalg.norm(center_pt - left_pt_cd1)
        dist2 = np.linalg.norm(center_pt - left_pt_cd2)
        if dist1 > dist2:
            offset_pt_l = left_pt_cd1
            offset_pt_r = right_pt_cd1
        else:
            offset_pt_l = left_pt_cd2
            offset_pt_r = right_pt_cd2
        return offset_pt_l, offset_pt_r

    def _vertices_from_edge_pts(self, robo_pos, edge_pos, edge_vel, const, gp_offset, offset=0):
        # Get the vertices of the pentagon social group from the edge points
        # Inputs:
        # robo_pos: the robot position
        # edge_pos: the edge position 3Nx2
        # edge_vel: the edge velocity 3Nx2
        # const: the constant for calculating the boundary distance
        # gp_offset: the length of the offset for the back edges to form pentagons
        # offset: the offset to be reducted from the boundary distance

        # Expand and account for personal spaces
        robo_pos = np.array(robo_pos)
        vertices = []
        num_pts = np.shape(edge_pos)[0]
        if not ((num_pts % 3) == 0):
            self.logger.error("num_pts not a multiplier of 3!")
            raise Exception("num_pts not a multiplier of 3!")
        # left first, close second, right third
        for i in range(num_pts):
            pos = edge_pos[i]
            vel = edge_vel[i]
            candidate_vts = np.array(draw_social_shapes([pos], [vel], const, offset))
            if (i % 3) == 1:
                dists = np.linalg.norm(candidate_vts - robo_pos, axis=1)
                min_idx = np.argmin(dists)
                pt_choice = candidate_vts[min_idx]
            else:
                pt_choice = None
                left_idx, right_idx = self._find_left_right_edge(np.array(candidate_vts), robo_pos)
                if (i % 3) == 0:
                    pt_choice = candidate_vts[left_idx]
                elif (i % 3) == 2:
                    pt_choice = candidate_vts[right_idx]
                else:
                    self.logger.error("i mod 3 cannot be 1 here!")
                    raise Exception("i mod 3 cannot be 1 here!")
                pt_choice = np.array([pt_choice[0], pt_choice[1]])
            vertices.append(pt_choice)

        # add offsets
        expanded_vertices = []
        num_pts = int(num_pts / 3)
        epsilon = 1e-5
        for i in range(num_pts):
            left_pt = vertices[i * 3]
            center_pt = vertices[i * 3 + 1]
            right_pt = vertices[i * 3 + 2]
            if ((np.linalg.norm(center_pt - left_pt) < epsilon) or
            (np.linalg.norm(center_pt - right_pt) < epsilon)):
                center_pt = (left_pt + right_pt) / 2
            left_offset_pt, right_offset_pt = self._construct_offset(left_pt, right_pt, robo_pos, gp_offset)
            expanded_vertices.append(left_pt)
            expanded_vertices.append(center_pt)
            expanded_vertices.append(right_pt)
            expanded_vertices.append(left_offset_pt)
            expanded_vertices.append(right_offset_pt)
        return np.array(expanded_vertices)

    def get_state_and_predictions(self, obs):
        # Get predictions for MPC
        # Linearly predict the future positions
        if self.laser:
            curr_pos = obs['laser_pos']
            group_ids = obs['laser_group_labels_history']
            history_pos = obs['laser_pos_history']
            history_vel = obs['laser_vel_history']
        else:
            curr_pos = obs['pedestrians_pos']
            group_ids = obs['group_labels']
            history_pos = obs['pedestrians_pos_history']
            history_vel = obs['pedestrians_vel_history']
        robot_pos = obs['robot_pos']

        self.dataset_info = obs['dataset_info']
        self.boundary_const = obs['personal_size']
        num_ped = len(curr_pos)

        self.edge_predictions = []

        if not num_ped == 0:
            if self.laser:
                edge_pos, _ = self.id_edge_pts_laser(robot_pos, history_pos, history_vel, group_ids, self.dt)
            else:
                edge_pos, _ = self.id_edge_pts(robot_pos, history_pos, history_vel, group_ids)
            future_edge_pos = self.sgan.evaluate(edge_pos)
            future_edge_vel = np.zeros_like(future_edge_pos)
            future_edge_vel[:, 1:, :] = (future_edge_pos[:, 1:, :] - future_edge_pos[:, :-1, :]) / self.dt
            future_edge_vel[:, 0, :] = (future_edge_pos[:, 0, :] - edge_pos[:, -1, :]) / self.dt

            assert np.shape(future_edge_pos)[1] == self.future_steps

            for i in range(self.future_steps):
                # result is a Tx5Nx2 array. Each 5x2 array is [left, center, right, left_offset, right_offset]
                # N is the number of groups
                self.edge_predictions.append(self._vertices_from_edge_pts(robot_pos, 
                                                                             future_edge_pos[:, i, :], 
                                                                             future_edge_vel[:, i, :], 
                                                                             self.boundary_const,
                                                                             self.edge_offset,
                                                                             self.offset))

        return
    
    def _dist_pt_line(self, p1, p2, target_pt):
        # Calculate the distance between a point and a line
        # Line is defined by two points p1 and p2

        dp = p2 - p1
        st = dp[0] ** 2 + dp[1] ** 2
        u = ((target_pt[0] - p1[0]) * dp[0] + (target_pt[1] - p1[1]) * dp[1]) / st
        if u > 1:
            u = 1
        elif u < 0:
            u = 0
        dx = (p1[0] + u * dp[0]) - target_pt[0]
        dy = (p1[1] + u * dp[1]) - target_pt[1]
        return np.sqrt(dx ** 2 + dy ** 2)
    
    def _find_least_dist_edges(self, target_pt, key_pts, shape):
        # Finds the least distance between a point and a set of edges

        num_pts = np.shape(key_pts)[0]
        if not ((num_pts % 5) == 0):
            raise Exception("num_pts not a multiplier of 5!")
        num_pts = int(num_pts / 5)
        min_dist = np.inf
        for i in range(num_pts):
            left_pt = key_pts[i * 5]
            center_pt = key_pts[i * 5 + 1]
            right_pt = key_pts[i * 5 + 2]
            left_offset_pt = key_pts[i * 5 + 3]
            right_offset_pt = key_pts[i * 5 + 4]
            if shape == "line":
                dist1 = self._dist_pt_line(center_pt, left_pt, target_pt)
                dist2 = self._dist_pt_line(left_pt, left_offset_pt, target_pt)
                dist3 = self._dist_pt_line(left_offset_pt, right_offset_pt, target_pt)
                dist4 = self._dist_pt_line(right_offset_pt, right_pt, target_pt)
                dist5 = self._dist_pt_line(right_pt, center_pt,  target_pt)
                min_dist_test = np.min([dist1, dist2, dist3, dist4, dist5])
                if min_dist_test < min_dist:
                    min_dist = min_dist_test
            else:
                self.logger.error("Shape not defined!")
                raise Exception("Shape not defined!")

        return min_dist
    
    def _point_in_pentagon(self, point, pentagon):
        # Check if a point is inside a pentagon
        # Inputs:
        # point: the point to be checked
        # pentagon: the vertices of the pentagon in order

        x, y = point
        inside = False
        n = len(pentagon)  # should be 5 for a pentagon

        assert n == 5

        for i in range(n):
            j = (i + 1) % n  # next vertex, wrapping around at the end
            xi, yi = pentagon[i]
            xj, yj = pentagon[j]
            
            # Check if the point is between the y-coordinates of the edge
            # and if it lies to the left of the line segment.
            if ((yi > y) != (yj > y)):
                # Compute the x-coordinate where the line from the point crosses the edge.
                intersect_x = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < intersect_x:
                    inside = not inside
        return inside
    
    def _check_inside_edges(self, target_pt, key_pts, shape):
        # Check if a point is inside the edges
        # Currently only supports pentagon shape
        num_pts = np.shape(key_pts)[0]
        if not ((num_pts % 5) == 0):
            raise Exception("num_pts not a multiplier of 5!")
        num_pts = int(num_pts / 5)
        for i in range(num_pts):
            left_pt = key_pts[i * 5]
            center_pt = key_pts[i * 5 + 1]
            right_pt = key_pts[i * 5 + 2]
            left_offset_pt = key_pts[i * 5 + 3]
            right_offset_pt = key_pts[i * 5 + 4]
            if shape == "line":
                if self._point_in_pentagon(target_pt, [left_pt, 
                                                       center_pt, 
                                                       right_pt, 
                                                       right_offset_pt, 
                                                       left_offset_pt]):
                    return True
            else:
                self.logger.error("Shape not defined!")
                raise Exception("Shape not defined!")
        return False
    
    def _rollout_dist(self, rollout, edge_pts):
        shape = "line"

        time_steps = np.shape(rollout)[0]
        dists = np.ones(time_steps)*(1e+9)
        hit_idx = time_steps
        for i in range(time_steps):
            if self._check_inside_edges(rollout[i], edge_pts[i], shape):
                hit_idx = min(hit_idx, i)
            dists[i] = self._find_least_dist_edges(rollout[i], edge_pts[i], shape)
        return dists, hit_idx
    
    def _min_dist_cost_func(self, dists, hit_idx):
        cost = 0
        gamma = self.gamma
        discount = 1
        for i, d in enumerate(dists):
            if i >= hit_idx:
                d = -d
            #cost += np.exp(-d)
            cost += np.exp(-d) * discount
            discount *= gamma
        return cost
    
    def evaluate_rollouts(self, mpc_weight=None):
        # Evaluate rollouts for MPC
        # Rollouts are NxTx2 arrays, where N is the number of rollouts, T is the number of time steps
        # Predictions are MxTx2 arrays, where M is the number of pedesrtians, T is the number of time steps

        if self.rollouts is None or self.edge_predictions is None:
            self.logger.error('Rollouts or predictions are not generated')
            raise ValueError('Rollouts or predictions are not generated')
        
        if self.dataset_info is None:
            self.logger.error('Dataset information is not set')
            raise ValueError('Dataset information is not set')
        
        if mpc_weight is None:
            mpc_weight = self.dist_weight

        if len(self.edge_predictions) == 0:
            has_ped = False
        else:
            has_ped = True

        self.rollout_costs = np.zeros(self.num_rollouts, dtype=np.float32)
        min_dist_weight = mpc_weight
        end_dist_weight = 1 - min_dist_weight # currently only 2 cost terms

        for i in range(self.num_rollouts):
            # Calculate the distance between the rollouts and predictions
            if has_ped:
                min_dists, hit_idx = self._rollout_dist(self.rollouts[i], self.edge_predictions)
                min_dist_cost = self._min_dist_cost_func(min_dists, hit_idx)
            else:
                min_dist_cost = 0
                hit_idx = self.future_steps
            if hit_idx == 0:
                end_dist_cost = np.linalg.norm(self.robot_goal - self.robot_pos)
            else:
                end_dist_cost = np.linalg.norm(self.robot_goal - self.rollouts[i, hit_idx - 1])
            self.rollout_costs[i] = min_dist_weight * min_dist_cost + end_dist_weight * end_dist_cost
        return