import json
import time
from pathlib import Path

import mujoco as mj
import mujoco.viewer

BASE_DIR = Path(__file__).resolve().parent
SCENE_PATH = BASE_DIR / "model" / "scene.xml"
ROPE_POSE_PATH = BASE_DIR / "rope_pose.jsonl"

model = mj.MjModel.from_xml_path(str(SCENE_PATH))
data = mj.MjData(model)

N = 12
rope_body_ids = []
for i in range(model.nbody):
    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
    if name and name.startswith("actuatedB_"):
        rope_body_ids.append(i)

record_interval = 1  # seconds
last_record_time = -record_interval


with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():
        step_start = time.time()

        # step simulation
        mj.mj_step(model, data)

        # --- RECORD EVERY 1 SECOND ---
        if data.time - last_record_time >= record_interval:
            last_record_time = data.time

            rope_state = []
            for i in rope_body_ids:
                rope_state.append(
                    {"pos": data.xpos[i].tolist(), "quat": data.xquat[i].tolist()}
                )
                # for quat: [w, x, y, z]

            with ROPE_POSE_PATH.open("a") as f:
                json.dump({"time": data.time, "rope": rope_state}, f)
                f.write("\n")

        v.sync()

        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
