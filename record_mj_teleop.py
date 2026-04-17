from pathlib import Path
import time

import mujoco as mj
import mujoco.viewer
import numpy as np
import mediapy as media


BASE_DIR = Path(__file__).resolve().parent
SCENE_PATH = BASE_DIR / "model" / "scene.xml"

VIDEO_PATH = BASE_DIR / "data" / "teleop_front_camera.mp4"
TRACKS_PATH = BASE_DIR / "data" / "teleop_rope_tracks_txy.npy"
OCCLUSION_PATH = BASE_DIR / "data" / "teleop_rope_tracks_occluded.npy"

CAMERA_NAME = "front_camera"

FPS = 24
DURATION = 10  # seconds
N_TRACKED_CAPSULES = 34
WIDTH = 1280
HEIGHT = 720


model = mj.MjModel.from_xml_path(str(SCENE_PATH))
data = mj.MjData(model)

rope_geom_ids = []
for i in range(model.ngeom):
    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
    if name and name.startswith("actuatedG"):
        rope_geom_ids.append(i)

start = (len(rope_geom_ids) - N_TRACKED_CAPSULES) // 2
# tracked_geom_ids = rope_geom_ids[start : start + N_TRACKED_CAPSULES]
tracked_geom_ids = [rope_geom_ids[i] for i in range(start, start + N_TRACKED_CAPSULES, 2)]

camera_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
camera_resolution = model.cam_resolution[camera_id]
camera_width = int(camera_resolution[0]) if camera_resolution[0] > 0 else WIDTH
camera_height = int(camera_resolution[1]) if camera_resolution[1] > 0 else HEIGHT

model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), camera_width)
model.vis.global_.offheight = max(int(model.vis.global_.offheight), camera_height)


# taken from the mujoco tutorial notebook and modified
def compute_camera_matrix(renderer, data):
    renderer.update_scene(data, CAMERA_NAME)
    # cam = renderer.scene.camera[0]
    # pos = cam.pos
    pos = data.cam_xpos[camera_id]
    # z = -cam.forward
    # y = cam.up
    # rot = np.vstack((np.cross(y, z), y, z))
    rot = data.cam_xmat[camera_id].reshape(3, 3).T
    fov = model.cam_fovy[camera_id]

    # Translation matrix (4x4).
    translation = np.eye(4)
    translation[0:3, 3] = -pos

    # Rotation matrix (4x4).
    rotation = np.eye(4)
    rotation[0:3, 0:3] = rot

    # Focal transformation matrix (3x4).
    focal_scaling = (1.0 / np.tan(np.deg2rad(fov) / 2)) * renderer.height / 2.0
    focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

    # Image matrix (3x3).
    image = np.eye(3)
    image[0, 2] = (renderer.width - 1) / 2.0
    image[1, 2] = (renderer.height - 1) / 2.0
    return image @ focal @ rotation @ translation


n_frames = int(FPS * DURATION)
capture_dt = 1.0 / FPS

tracks_txy = np.full((n_frames, N_TRACKED_CAPSULES, 3), -1.0, dtype=np.float32)
occluded = np.zeros((n_frames, N_TRACKED_CAPSULES), dtype=np.uint8)

options = mj.MjvOption()
mj.mjv_defaultOption(options)

mj.mj_resetData(model, data)
mj.mj_forward(model, data)

frames = []
next_capture_time = capture_dt
frame_idx = 0

with mj.Renderer(model, height=camera_height, width=camera_width) as renderer:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and frame_idx < n_frames:
            step_start = time.time()
            mj.mj_step(model, data)
            viewer.sync()

            while frame_idx < n_frames and data.time >= next_capture_time:
                renderer.update_scene(data, CAMERA_NAME)
                camera_matrix = compute_camera_matrix(renderer, data)
                frame = renderer.render()
                frames.append(frame)

                for i, geom_id in enumerate(tracked_geom_ids):
                    rope_pos = data.geom_xpos[geom_id].reshape(3, 1)
                    rope_pos_homogeneous = np.ones((4, 1))
                    rope_pos_homogeneous[:3] = rope_pos

                    xs, ys, s = camera_matrix @ rope_pos_homogeneous
                    x = int(np.round(xs / s)[0])
                    y = int(np.round(ys / s)[0])

                    x_for_select = int(np.clip(x, 0, camera_width - 1))
                    y_for_select = int(np.clip(y, 0, camera_height - 1))
                    sel_pos = np.zeros(3)
                    projected_geom_id = np.array([-1], dtype=np.int32)
                    flex_id = np.array([0], dtype=np.int32)
                    skin_id = np.array([0], dtype=np.int32)
                    mj.mjv_select(
                        model,
                        data,
                        options,
                        1.0,
                        x_for_select,
                        y_for_select,
                        renderer.scene,
                        sel_pos,
                        projected_geom_id,
                        flex_id,
                        skin_id,
                    )
                    selected_geom_id = int(projected_geom_id[0])
                    if selected_geom_id != geom_id:
                        occluded[frame_idx, i] = 1

                    tracks_txy[frame_idx, i, :] = np.array([frame_idx, x, y])

                frame_idx += 1
                next_capture_time += capture_dt

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)

np.save(TRACKS_PATH, tracks_txy)
np.save(OCCLUSION_PATH, occluded)
media.write_video(str(VIDEO_PATH), frames, fps=FPS)
