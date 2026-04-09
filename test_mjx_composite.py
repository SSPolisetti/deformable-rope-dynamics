from pathlib import Path

import mujoco
from mujoco import mjx


def main() -> None:
    scene_path = Path(__file__).resolve().parent / "model" / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))

    print(f"Loaded MuJoCo model from: {scene_path}")
    print(f"nbody={model.nbody}, njnt={model.njnt}, ngeom={model.ngeom}")

    mjx_model = mjx.put_model(model)

    print("MJX accepted the model.")
    print(f"mjx nq={mjx_model.nq}, nv={mjx_model.nv}, nu={mjx_model.nu}")


if __name__ == "__main__":
    main()
