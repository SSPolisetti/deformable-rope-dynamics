from pathlib import Path

import mediapy as media
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import cv2 as cv

from tapnet.tapnext.tapnext_torch import TAPNext, posemb_sincos_2d
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint
from tapnet.utils import viz_utils


bootstrapped = "boots"
BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "data" / "teleop_front_camera.mp4"
QUERY_TRACKS_PATH = BASE_DIR / "data" / "teleop_rope_tracks_txy.npy"
CKPT_PATH = BASE_DIR / "checkpoints" / f"{bootstrapped}tapnext_ckpt.npz"

OUTPUT_TRACKS_PATH = BASE_DIR / "data" / f"teleop_{bootstrapped}tapnext_pred_tracks_txy.npy"
OUTPUT_OCCLUDED_PATH = BASE_DIR / "data" / f"teleop_{bootstrapped}tapnext_pred_occluded.npy"
OUTPUT_VIDEO_PATH = BASE_DIR / "data" / f"teleop_{bootstrapped}tapnext_vis.mp4"


CKPT_SIZE = (256, 256)
SWAP_TRACK_XY = True
OUTPUT_FPS = 24



def run_online_tracking(
    model: TAPNext, video: torch.Tensor, query_points: torch.Tensor
):
    with torch.no_grad():
        pred_tracks, _, visible_logits, tracking_state = model(
            video=video[:, :1],
            query_points=query_points,
        )
        pred_tracks_list = [pred_tracks.cpu()]
        pred_visible_list = [(visible_logits > 0).cpu()]

        for frame_idx in tqdm.tqdm(range(1, video.shape[1])):
            curr_tracks, _, curr_visible_logits, tracking_state = model(
                video=video[:, frame_idx : frame_idx + 1],
                state=tracking_state,
            )
            pred_tracks_list.append(curr_tracks.cpu())
            pred_visible_list.append((curr_visible_logits > 0).cpu())

    tracks = torch.cat(pred_tracks_list, dim=1).transpose(1, 2).numpy()[0]
    visible = torch.cat(pred_visible_list, dim=1).transpose(1, 2).numpy()[0, :, :, 0]
    if SWAP_TRACK_XY:
        tracks = tracks[..., ::-1]
    occluded = ~visible
    return tracks, occluded


def main() -> None:
    print(f"Loading video: {VIDEO_PATH}")
    frames = cv.VideoCapture(str(VIDEO_PATH))

    np_frames = []
    i = 0
    while True:
        ret, frame = frames.read()
        if not ret:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        np_frames.append(frame)

    frames = np.array([np_frames])
    print(f"Loading query tracks: {QUERY_TRACKS_PATH}")
    query_tracks_txy = np.load(QUERY_TRACKS_PATH)
    first_entry_txy = query_tracks_txy[0]
    query_points_np = first_entry_txy[:, [0, 2, 1]].astype(np.float32)
    valid = np.logical_and(query_points_np[:, 1] >= 0, query_points_np[:, 2] >= 0)
    query_points_np = np.array([query_points_np[valid]]) # batch 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TAPNext(image_size=frames.shape[2:4])
    model = restore_model_from_jax_checkpoint(model, str(CKPT_PATH))
    # model = upgrade_to_target_resolution(model, MODEL_SIZE)
    model = model.to(device).eval()

    video = torch.from_numpy(frames.astype(np.float32))
    video = (video / 255.0) * 2.0 - 1.0
    query_points = torch.from_numpy(query_points_np)
    video = video.to(device)
    query_points = query_points.to(device)

    print("Running TAPNext tracking...")
    tracks, occluded = run_online_tracking(model, video, query_points)

    n_frames = tracks.shape[1]
    n_points = tracks.shape[0]
    pred_tracks_txy = np.zeros((n_frames, n_points, 3), dtype=np.float32)
    pred_tracks_txy[:, :, 0] = np.arange(n_frames, dtype=np.float32)[:, None]
    pred_tracks_txy[:, :, 1] = tracks[:, :, 0].T
    pred_tracks_txy[:, :, 2] = tracks[:, :, 1].T

    pred_occluded = occluded.T.astype(np.uint8)

    print(f"Saving predictions: {OUTPUT_TRACKS_PATH}")
    np.save(OUTPUT_TRACKS_PATH, pred_tracks_txy)
    print(f"Saving occlusions: {OUTPUT_OCCLUDED_PATH}")
    np.save(OUTPUT_OCCLUDED_PATH, pred_occluded)

    print("Rendering visualization...")
    painted = viz_utils.paint_point_track(
        frames=frames[0],
        point_tracks=tracks,
        visibles=~occluded,
    )
    print(f"Saving visualization video: {OUTPUT_VIDEO_PATH}")
    save = cv.VideoWriter(
        str(OUTPUT_VIDEO_PATH),
        cv.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_FPS,
        (painted.shape[2], painted.shape[1]),
    )

    if not save.isOpened():
        print("something wrong")
        return

    for i in range(painted.shape[0]):
        save.write(cv.cvtColor(painted[i], cv.COLOR_RGB2BGR))


    save.release()

    print("Done.")
    print(f"Tracked points: {n_points}, frames: {n_frames}")


if __name__ == "__main__":
    main()
