import pickle

with open("data/nuscenes/nuscenes_infos_trainval_with_inds.pkl", "rb") as f:
    content = pickle.load(f)
    input()

"""
lidar_path
token
sweeps
cams
lidar2ego_translation
lidar2ego_rotation
ego2global_translation
ego2global_rotation
timestamp
gt_boxes
gt_names
gt_velocity
num_lidar_pts
num_radar_pts
valid_flag
"""
