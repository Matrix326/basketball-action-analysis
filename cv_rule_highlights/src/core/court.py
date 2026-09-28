"""Coarse court target from existing cameras; never a scoring-plane calibration."""

import json
from pathlib import Path

import numpy as np
import yaml


def court_geometry(config_path, rims):
    config = yaml.safe_load(Path(config_path).read_text())
    def asset(value):
        return Path(value.replace('${project_root}', config['project_root']))
    camera = config['camera']
    intrinsics = json.loads(asset(camera['intrinsics_path']).read_text())
    extrinsics = json.loads(asset(camera['extrinsics_path']).read_text())
    projections, equations = {}, []
    for view, rim in rims.items():
        name = camera['view_to_camera'][view]
        intr, ext = intrinsics[name], extrinsics[name]
        k = np.asarray(intr.get('K_undistorted', intr.get('K_original')))
        rotation = np.asarray(ext.get('R_w2c', ext.get('R')))
        translation = np.asarray(ext.get('t_w2c', ext.get('t')))
        p = k @ np.column_stack((rotation, translation))
        projections[view] = p
        equations.extend((rim[0] * p[2] - p[0], rim[1] * p[2] - p[1]))
    _, _, vh = np.linalg.svd(equations)
    position = vh[-1, :3] / vh[-1, 3]
    errors = {}
    for view, p in projections.items():
        projected = p @ np.r_[position, 1]
        errors[view] = float(np.linalg.norm(projected[:2] / projected[2] - rims[view][:2]))
    return {'basket_ground_xy': position[:2].tolist(),
            'rim_reprojection_errors_px': errors,
            'usable_for_progress': all(error < .3 * rims[v][2] for v, error in errors.items()),
            'usable_for_scoring': False,
            'source': 'triangulated_configured_rim_centers_coarse_ground_target'}
