from pathlib import Path

from dm_control import mjcf
from mujoco import viewer

MODEL_DIR = Path(__file__).resolve().parent
ARM_XML = MODEL_DIR / "kuka_iiwa_14" / "iiwa14.xml"
GRIPPER_XML = MODEL_DIR / "robotiq_2f85_v4" / "2f85.xml"
ARM_SPACING = 1.0


class Arm:
    def __init__(self, arm_path, gripper_path, model = "arm_with_gripper"):
        self.mjcf_model = mjcf.from_path(ARM_XML)
        
        gripper_site = self.mjcf_model.find("site", "attachment_site")
        gripper_model = mjcf.from_path(GRIPPER_XML)

        self.gripper = gripper_site.attach(gripper_model)



left_arm = Arm(ARM_XML, GRIPPER_XML, "left_kuka_with_gripper")
right_arm = Arm(ARM_XML, GRIPPER_XML, "right_kuka_with_gripper")

model = mjcf.RootElement(model="dual_kuka_model")

left_arm_mount = model.worldbody.add("site", name="left_arm_mount", pos=(0.0, ARM_SPACING/ 2.0, 0.0))
right_arm_mount = model.worldbody.add("site", name="right_arm_mount", pos=(0.0, -ARM_SPACING/ 2.0, 0.0))

left_arm_mount.attach(left_arm.mjcf_model)
right_arm_mount.attach(right_arm.mjcf_model)

mjcf.export_with_assets(model, out_dir="dual_kuka_with_2f85", out_file_name="dual_kuka_with_2f85.xml")
