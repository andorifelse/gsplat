import torch
from gsplat import export_splats
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str, help="Path to the .pt checkpoint file")
    parser.add_argument("--output", type=str, default="output.ply", help="Output .ply file path")
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.ckpt_path}...")
    ckpt = torch.load(args.ckpt_path, map_location="cpu") # 加载到 CPU 即可
    splats = ckpt["splats"]

    print("Exporting to PLY...")
    export_splats(
        means=splats["means"],
        scales=splats["scales"],
        quats=splats["quats"],
        opacities=splats["opacities"],
        sh0=splats["sh0"],
        shN=splats["shN"],
        format="ply",
        save_to=args.output,
    )
    print(f"Done! Saved to {args.output}")

if __name__ == "__main__":
    main()