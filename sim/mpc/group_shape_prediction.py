import numpy as np
import torch

from sim.mpc.group import draw_social_shapes
from sim.mpc.img_process import ProcessImage, DrawGroupShape
from sim.mpc.conv_model import ConvAutoencoder

class GroupShapePrediction(object):

    def __init__(self, dataset_info, path, logger, device=0):
        # No need to do grouping here for dataset_info
        self.dataset_info = dataset_info

        if path is None:
            logger.warning('No model path provided. Only linear prediction is available!')
        else:
            self.cuda = torch.device('cuda:' + str(device))
            self.model = ConvAutoencoder()

            self.model.load_state_dict(torch.load(path, map_location='cpu'))
            self.model.eval()
            self.model.to(self.cuda)

        self.logger = logger
        self.logger.info('Model initialized!')

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

    def _predict_sequence(self, input_sequence):
        # Predict the output sequence from the input sequence using the model

        confidence_threshold = 0.5
        
        inputs = np.transpose(np.array(input_sequence), (3, 0, 1, 2))
        inputs_tensor = np.expand_dims(inputs, 0)
        inputs_tensor = torch.tensor(inputs_tensor, dtype=torch.float32, device=self.cuda)

        outputs_tensor = self.model(inputs_tensor)

        outputs = outputs_tensor.data.cpu().numpy()
        output_sequence = np.transpose(outputs[0, :, :, :, :], (1, 2, 3, 0))

        for i in range(len(output_sequence)):
            output_sequence[i] = np.round(output_sequence[i] >= confidence_threshold)
        return output_sequence

    def _predict_from_vertices(self, vertice_sequence):
        # Predict the group shape from the vertices
        # The input is a sequence of vertices

        dgs = DrawGroupShape(self.dataset_info)
        dgs.set_center(vertice_sequence)
        dgs.set_aug(angle=0)
        img_sequence = []
        for i, v in enumerate(vertice_sequence):
            canvas = np.zeros((self.dataset_info.frame_height, self.dataset_info.frame_width, 3), dtype=np.uint8)
            img = dgs.draw_group_shape(v, canvas, center=True, aug=False)
            img_sequence.append(img)

        # Process the image
        pimg = ProcessImage(self.dataset_info, img_sequence)
        for i, img in enumerate(img_sequence):
            img_sequence[i] = pimg.process_image(img, debug=False)

        pred_img_sequence = self._predict_sequence(img_sequence)

        # Reverse the process
        group_pred_img_sequence = []
        for i, img in enumerate(pred_img_sequence):
            img = np.round(np.repeat(img, 3, axis=2))
            pred_img = pimg.reverse_process_image(img, debug=True)
            pred_img = dgs.reverse_move_center_img(pred_img)
            group_pred_img_sequence.append(pred_img[:, :, 0])
        return group_pred_img_sequence

    def _compile_group_pred(self, all_pred_img_sequences, num_groups):
        # Compile the prediction of each group into a single prediction
        # by summing up the predictions of each group into a single image

        fnl_pred_img_sequence = []
        if len(all_pred_img_sequences) == 0:
            return fnl_pred_img_sequence
        pred_length = len(all_pred_img_sequences[0])
        for i in range(pred_length):
            canvas = np.zeros((self.dataset_info.frame_height, self.dataset_info.frame_width), dtype=np.uint8)
            for j in range(num_groups):
                img = all_pred_img_sequences[j][i]
                img = np.round(img)
                canvas += img
            fnl_pred_img_sequence.append(np.clip(canvas, 0, 1))
        return fnl_pred_img_sequence

    def predict(self, positions, velocities, labels, const, offset=0):
        # Predict the group shapes using the positions and velocities of the pedestrians
        # Inputs:
        # positions: the positions of the pedestrians NxTx2
        # velocities: the velocities of the pedestrians NxTx2
        # labels: the group labels of the pedestrians at the current time N
        # const: the constant for calculating the boundary distance
        # offset: the offset to be reducted from the boundary distance
        # Outputs:
        # group_pred_img_sequence: the predicted group shapes

        num_people = len(positions)

        if num_people == 0:
            self.logger.error('No people detected!')
            raise Exception('No people detected!')

        seq_length = len(positions[0])

        all_labels = np.unique(labels)
        num_groups = len(all_labels)
        all_pred_img_sequences = []

        for curr_label in all_labels:
            # Get the positions and velocities of the group members
            group_positions = []
            group_velocities = []
            for i, l in enumerate(labels):
                if l == curr_label:
                    group_positions.append(positions[i])
                    group_velocities.append(velocities[i])
            group_positions = np.array(group_positions)
            group_velocities = np.array(group_velocities)
            
            # Get the vertices of the group shape
            vertice_sequence = []
            for i in range(seq_length):
                frame_positions = group_positions[:, i, :]
                frame_velocities = group_velocities[:, i, :]
                vertices = draw_social_shapes(frame_positions, frame_velocities, const, offset)
                vertice_sequence.append(vertices)

            group_pred_img_sequence = self._predict_from_vertices(vertice_sequence)
            all_pred_img_sequences.append(group_pred_img_sequence)

        return self._compile_group_pred(all_pred_img_sequences, num_groups)

    def laser_predict(self, positions, velocities, group_ids, dt, const, offset=0):
        # Predict the group shapes using the laser data
        # Inputs:
        # positions: the positions of the laser points Tx?x2, ? = non-fixed
        # velocities: the velocities of the laser points Tx?x2
        # group_ids: the group ids of the laser points Tx?
        # dt: the time interval between frames
        # const: the constant for calculating the boundary distance
        # offset: the offset to be reducted from the boundary distance
        # Outputs:
        # group_pred_img_sequence: the predicted group shapes

        # Nearest geo-center way of building history
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
                group_positions = []
                group_velocities = []
                center_x = 0
                center_y = 0
                center_vx = 0
                center_vy = 0
                for k, l in enumerate(labels):
                    if curr_label == l:
                        group_positions.append(pos[k])
                        group_velocities.append(vel[k])
                        center_x += pos[k][0]
                        center_y += pos[k][1]
                        center_vx += vel[k][0]
                        center_vy += vel[k][1]
                all_group_pos.append(group_positions)
                all_group_vel.append(group_velocities)
                num_members = len(group_positions)
                center_x /= num_members
                center_y /= num_members
                center_vx /= num_members
                center_vy /= num_members
                centers.append(np.array([center_x, center_y]))
                vel_centers.append(np.array([center_vx, center_vy]))
            group_pos_series.append(all_group_pos)
            group_vel_series.append(all_group_vel)
            group_centers.append(centers)
            group_vel_centers.append(vel_centers)

        temp_threshold = 2.5 * dt #m/s / fps
        num_curr_groups = len(group_pos_series[-1])
        all_pred_img_sequences = []
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

            vertice_sequence = []
            for j in range(time_steps-1, -1, -1):
                vertices = draw_social_shapes(position_seq[j], velocity_seq[j], const, offset)
                vertice_sequence.append(vertices)

            group_pred_img_sequence = self._predict_from_vertices(vertice_sequence)
            all_pred_img_sequences.append(group_pred_img_sequence)

        return self._compile_group_pred(all_pred_img_sequences, num_curr_groups)

    # Linear prediction
    def linear_predict(self, positions, velocities, labels, pred_seq_length, dt, const, offset=0):
        # Linearly predict the future positions and velocities of the groups
        # by using the current central positions and velocities of the groups
        # Inputs:
        # positions: the current positions of the pedestrians Nx2
        # velocities: the current velocities of the pedestrians Nx2
        # labels: the group labels of the pedestrians at the current time N
        # pred_seq_length: the length of the prediction sequence
        # dt: the time interval between frames
        # const: the constant for calculating the boundary distance
        # offset: the offset to be reducted from the boundary distance
        # Outputs:
        # group_pred_img_sequence: the predicted group shapes

        all_labels = np.unique(labels)
        num_groups = len(all_labels)
        all_pred_img_sequences = []
        for i, curr_label in enumerate(all_labels):
            group_positions = []
            group_velocities = []
            center_vx = 0
            center_vy = 0
            for j, l in enumerate(labels):
                if curr_label == l:
                    group_positions.append(positions[j])
                    group_velocities.append(velocities[j])
                    center_vx += velocities[j][0]
                    center_vy += velocities[j][1]
            num_members = len(group_positions)
            center_vx /= num_members
            center_vy /= num_members
            vel_center = np.array([center_vx, center_vy])

            group_positions = np.array(group_positions)
            group_velocities = np.array(group_velocities)
            vertice_sequence = []
            for j in range(pred_seq_length):
                future_positions = group_positions + j * vel_center * dt
                vertices = draw_social_shapes(future_positions, group_velocities, const, offset)
                vertice_sequence.append(vertices)

            dgs = DrawGroupShape(self.dataset_info)
            img_sequence = []
            for j, v in enumerate(vertice_sequence):
                canvas = np.zeros((self.dataset_info.frame_height, self.dataset_info.frame_width, 3), dtype=np.uint8)
                img = dgs.draw_group_shape(v, canvas, center=False, aug=False)
                img_sequence.append(img[:, :, 0])
            all_pred_img_sequences.append(img_sequence)

        return self._compile_group_pred(all_pred_img_sequences, num_groups)
