import argparse
import os

from src.io import *
from src.model import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the garment deformation model"
    )

    parser.add_argument(
        "motion_path",
        type=str,
        help="path to the .npz file with the motion data"
    )

    parser.add_argument(
        "--export_dir",
        type=str,
        default="results",
        help="directory to save the predictions"
    )

    args = parser.parse_args()

    motion = load_motion(args.motion_path)
    v_garment, v_body = run_model(motion)

    # Save meshes
    os.makedirs(args.export_dir, exist_ok=True)

    _, f_garment = load_obj("assets/meshes/tshirt.obj")
    _, f_body = load_obj("assets/meshes/body.obj")

    for i in range(len(v_garment)):
        path = os.path.join(args.export_dir, f"{i:04d}_body.obj")
        save_obj(path, v_body[i], f_body)

        path = os.path.join(args.export_dir, f"{i:04d}_garment.obj")
        save_obj(path, v_garment[i], f_garment)

