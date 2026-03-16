import time

import mujoco as mj
import mujoco.viewer

model = mj.MjModel.from_xml_path("./model/dual_iiwa14_2f85/scene.xml")
data = mj.MjData(model)

with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():
        step_start = time.time()

        # Advance simulation by one step
        mj.mj_step(model, data)

        # Update viewer
        v.sync()

        # Keep real-time pacing
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
