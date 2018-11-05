import numpy as np
import cv2
import torch
from torch.multiprocessing import Pool, Process, set_start_method
import itertools
import matplotlib.pyplot as plt
import torch.multiprocessing as mp


def cut_empty(img, padding=30):
    rows, cols = img.shape
    e_r = img.max(1).nonzero()[0]
    min_non_e_r = e_r[0]
    max_non_e_r = e_r[-1]
    e_c = img.max(0).nonzero()[0]
    min_non_e_c = e_c[0]
    max_non_e_c = e_c[-1]

    crop_rows = (max(0, min_non_e_r - padding), min(rows, max_non_e_r + padding))
    crop_cols = (max(0, min_non_e_c - padding), min(cols, max_non_e_c + padding))
    r_img = img[crop_rows[0]: crop_rows[1], crop_cols[0]: crop_cols[1]]
    return r_img, (crop_rows, crop_cols)


def overlap_score(args):
    """
    :param full_map:
    :param point_cloud:
    :param rotation_orig: (row, col) of point cloud
    :param pos: (row, col)
    :param orientation: degrees
    :return:
    """
    full_map, point_cloud, rotation_orig, pos, orientation, max_dim = args

    point_cloud = point_cloud.numpy()
    rows, cols = point_cloud.shape

    # Rotate crop from map
    out_d = max_dim * 2 + 1
    map_overlap = full_map[
                  pos[0] - max_dim: pos[0] + max_dim + 1,
                  pos[1] - max_dim: pos[1] + max_dim + 1]

    map_overlap = map_overlap.numpy()
    rotation_m = cv2.getRotationMatrix2D((max_dim, max_dim), orientation, 1.)
    map_overlap = cv2.warpAffine(map_overlap, rotation_m, (out_d, out_d))

    # Select point cloud intersection zone
    start_row = max_dim - rotation_orig[0]
    start_col = max_dim - rotation_orig[1]
    map_overlap = map_overlap[start_row: start_row + rows, start_col: start_col + cols]

    # img = map_overlap
    # # img = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
    # cv2.imshow("test", img*255  )
    #
    # img = dst
    # # img = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
    # cv2.imshow("test2", img*255  )
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Multiply intensities of overlapping pixels
    # overlap_score = 0
    overlap_score = (point_cloud * map_overlap).sum()

    # # Subtract not matched pixels from map
    # overlap_score -= map_overlap[dst == 0].sum()

    return overlap_score

def overlap_score_test(args):
    return 0


class LocateMap:
    def __init__(self, map_cfg, localization_cfg):
        occupancy_map = map_cfg.occupancy_map
        semantic_map = map_cfg.semantic_map
        lane_map = map_cfg.lane_map
        scale_px_cm = map_cfg.scale_px_cm
        map_labels = map_cfg.map_labels

        # Transform variables
        map_labels = dict({x: np.array(y, dtype=np.uint8) for x, y in map_labels})

        # Localization cfg
        self.normal_orientation = localization_cfg.normal_orientation
        self.no_particles = localization_cfg.no_particles
        self.no_workers = no_workers = localization_cfg.no_workers

        # Should be with 0 - Free space/ 255 occupancy
        occupancy_map = cv2.imread(occupancy_map, cv2.IMREAD_UNCHANGED)
        semantic_map = cv2.imread(semantic_map)
        lane_map = cv2.imread(lane_map, cv2.IMREAD_UNCHANGED)

        self.road_map = ((semantic_map == map_labels["road"]).sum(2) == 3)

        assert len(lane_map.shape) == 3, "Wrong shape for lane_map"

        lane_map = cv2.cvtColor(lane_map, cv2.COLOR_BGR2GRAY)
        # lane_map = lane_map.astype(np.float32)

        self.shared_lane_map = torch.from_numpy(lane_map)
        # self.shared_lane_map.div_(255.)
        self.shared_lane_map.share_memory_()

        self.point_cloud_mesh_size = None
        self.point_cloud_mesh = None

        self.log = True

    def locate(self, point_cloud_view, p_pos, p_orientation, pos_std, orientation_std):
        """
        :param point_cloud_view: Point_cloud_img_view (Scaled to map)
        :param p_pos: Predicted location position Pixels (row, col)
        :param p_orientation: The heading in degrees relative to the geographic North Pole.
        :param pos_acc: Pos acc in px
        :param orientation_acc: in degrees
        :return:
        """
        # # -- Test data
        # point_cloud_view = gt_port_img
        # p_pos = data_pos
        # p_orientation = data_orientation
        #
        # pos_std = sim_noise_pos
        # orientation_std = sim_noise_direction
        # # -----------------------------------

        no_particles = self.no_particles
        road_map = self.road_map
        normal_orientation = self.normal_orientation
        point_cloud_mesh_size = self.point_cloud_mesh_size
        shared_lane_map = self.shared_lane_map

        # -- Generate Positions & orientations
        x_offset = np.random.normal(0, pos_std, no_particles).astype(np.int)
        y_offset = np.random.normal(0, pos_std, no_particles).astype(np.int)

        possible_x = x_offset + p_pos[0]
        possible_y = y_offset + p_pos[1]

        # -- Filter points positions
        # on_road = road_map[possible_x, possible_y].nonzero()[0]
        # possible_pos = list(zip(possible_x[on_road], possible_y[on_road]))
        possible_pos = list(zip(possible_x, possible_y))

        p_orientation = p_orientation - normal_orientation  # adjust for std orientation

        possible_o = p_orientation + np.random.normal(0, orientation_std, len(possible_pos))

        # Fill map for rotation ->
        rows_p, cols_p = point_cloud_view.shape
        max_dim = int(np.sqrt(rows_p ** 2 + cols_p ** 2) + 1)

        # # center to ->
        # r_start = max_dim - rows_p // 2
        point_cloud_mesh = torch.from_numpy(point_cloud_view)
        point_cloud_mesh.share_memory_()
        center_rotation = (rows_p // 2, 0)
        # point_cloud_mesh[r_start: r_start + rows_p, max_dim: max_dim + cols_p] = point_cloud_view
        point_cloud_mesh.div_(255.)

        # get pool
        args_list = zip(itertools.cycle([shared_lane_map]),
                        itertools.cycle([point_cloud_mesh]),
                        itertools.cycle([center_rotation]),
                        possible_pos, possible_o,
                        itertools.cycle([max_dim]))

        # Seems slow :(
        # multi_pool = Pool(processes=self.no_workers)
        # predictions = multi_pool.map(overlap_score, args_list)
        # multi_pool.close()
        # multi_pool.join()

        predictions = [overlap_score(args) for args in args_list]

        predictions = torch.tensor(predictions, dtype=torch.float32)
        predictions = (predictions - predictions.min())
        predictions = predictions / predictions.max()

        # predictions = torch.nn.Softmax()(predictions)

        if self.log:
            # View prob map
            lane_map_view = (shared_lane_map * 255.).numpy().astype(np.uint8)
            cv2.imwrite("test2.png", lane_map_view)
            lane_map_view = cv2.cvtColor(lane_map_view, cv2.COLOR_GRAY2BGR)

            global ground_pos

            dist_to_ground = []
            possible_pos = np.array(possible_pos)
            for ix, (x, y) in enumerate(possible_pos):
                d = np.linalg.norm(np.array([x, y]) - np.array(ground_pos))
                dist_to_ground.append(d)
                lane_map_view[x, y, :2] = 0
                lane_map_view[x, y, 2] += int(predictions[ix] * 50.)
                # print(f"Dist: {d} _ prob: {predictions[ix]}")

            # crop zone around
            lane_map_view = lane_map_view[p_pos[0]-max_dim: p_pos[0]+max_dim,
                            p_pos[1] - max_dim: p_pos[1] + max_dim ]

            select = predictions.sort()[1][-3:]
            plt.scatter(possible_pos[select, 1], possible_pos[select, 0], s=predictions[select]*10)
            plt.scatter([ground_pos[1]], [ground_pos[0]])
            plt.show()

            # for x, y in possible_pos:
            #     occupancy_map_view[x, y
            # img = ((occupancy_map > 0) * 255).astype(np.uint8)
            img = lane_map_view
            # img = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
            cv2.imshow("test", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


class TestPointCloud:
    def __init__(self, full_cfg):
        test_cfg = full_cfg.test
        self.sim_noise_pos = test_cfg.sim_noise_pos
        self.sim_noise_direction = test_cfg.sim_noise_direction
        self.accuracy = test_cfg.accuracy
        self.scale_px_cm = full_cfg.map.scale_px_cm
        ground_truth_point_path = test_cfg.ground_truth_point_path

        ground_truth_point_path_img = ground_truth_point_path + ".png"
        ground_truth_point_path_corners = ground_truth_point_path + ".log"

        point_cloud = cv2.imread(ground_truth_point_path_img, cv2.IMREAD_UNCHANGED)
        point_cloud = ((point_cloud.sum(2) > 0) * 255).astype(np.uint8)  # Reduce to 1 channel
        self.point_cloud = point_cloud

        with open(ground_truth_point_path_corners, "r") as f:
            ground_pos = eval(f.readline())
            ground_orientation = eval(f.readline())

        # center left corner
        self.ground_pos = ground_pos
        self.ground_orientation = ground_orientation

        # rotation center
        rows, cols = point_cloud.shape
        self.max_dim = int(np.sqrt(rows**2 + cols**2) + 1)
        self.ground_truth_pos = [rows//2, 0]

        # img = occupancy_map[corners[0][0]:corners[1][0], corners[0][1]:corners[1][1]]

    def get_test(self):
        sim_noise_pos = self.sim_noise_pos / self.scale_px_cm
        sim_noise_direction = self.sim_noise_direction
        ground_pos = self.ground_pos
        ground_orientation = self.ground_orientation
        point_cloud = self.point_cloud
        max_dim = self.max_dim

        # ADD space for rotation
        rows, cols = point_cloud.shape
        new_demo_size = (max_dim*2, max_dim*2)
        demo = np.zeros(new_demo_size, np.uint8)

        # center to ->
        r_start = max_dim - rows//2
        demo[r_start: r_start+rows, max_dim: max_dim+cols] = point_cloud
        center_rotation = (max_dim, max_dim)

        # TODO calculate another ground truth viewport

        # # Rotate
        #
        # conv_matrix = cv2.getRotationMatrix2D(center_rotation, error_angle, 1)
        # dst = cv2.warpAffine(demo, conv_matrix, new_demo_size)
        #
        # # Crop & recalculate ground truth point (rows/2, 0)
        # crop, margins = cut_empty(dst)

        # cv2.imshow("test", crop)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        gt_port_img = point_cloud
        gt_port_pos = ground_pos
        gt_port_orientation = ground_orientation

        # Fake gps & orientation
        error_pos = np.random.uniform(-sim_noise_pos, sim_noise_pos, 2).astype(np.int)
        x = ground_pos[0] + error_pos[0]
        y = ground_pos[1] + error_pos[1]

        error_angle = np.random.uniform(-sim_noise_direction, sim_noise_direction, 1)[0]

        data_orientation = ground_orientation + error_angle
        data_pos = [x, y]

        return (gt_port_img,
                [data_pos, data_orientation],
                [gt_port_pos, gt_port_orientation],
                [sim_noise_pos, sim_noise_direction])


if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    from utils import read_cfg
    import time

    config_file = "configs/default.yaml"
    full_cfg = read_cfg(config_file)
    map_cfg = full_cfg.map
    localization_cfg = full_cfg.localization

    test_generator = TestPointCloud(full_cfg)
    (gt_port_img,
     [data_pos, data_orientation],
     [gt_port_pos, gt_port_orientation],
     [sim_noise_pos, sim_noise_direction]) = test_generator.get_test()

    global ground_pos
    ground_pos = gt_port_pos

    print(sim_noise_pos)

    localization = LocateMap(map_cfg, localization_cfg)

    t = []
    for i in range(30):
        st = time.time()
        localization.locate(gt_port_img, data_pos, data_orientation, sim_noise_pos,
                            sim_noise_direction)
        t.append(time.time() - st)
        print(t[-1])
    print("---mean---")
    print(np.mean(t))

    # # imshow
    # img = ((occupancy_map > 0) * 255).astype(np.uint8)
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # cv2.imshow("test", r_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()