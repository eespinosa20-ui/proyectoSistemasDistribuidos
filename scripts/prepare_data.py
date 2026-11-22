import argparse
import csv
from pathlib import Path

import numpy as np

FIELDS = ("scene_token", "split", "category", "width", "length", "height",
          "distance", "visibility", "num_lidar_pts", "num_radar_pts", "velocity")
CLASSES = ("vehiculo", "peaton", "otros")


def partition(index):
    return ("train", "val", "test")[(index >= 6) + (index >= 8)]


def mock():
    rng = np.random.default_rng(42)
    for index in range(10):
        for sample in range(12):
            label = sample % 3
            width, length, height = rng.uniform(0.5, 4, 3)
            distance = rng.uniform(1, 50)
            points = max(0, round(80 + 8 * width + 4 * length - 1.6 * distance + rng.normal()))
            row = (f"mock-{index:02}", partition(index), CLASSES[label], width, length,
                   height, distance, sample % 4 + 1, points, sample % 5, rng.uniform(0, 10))
            image = rng.integers(0, 48, (64, 64, 3), dtype=np.uint8)
            image[:, :, label] += 170
            yield row, [(image, label)]


def nuscenes(root):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import BoxVisibility, view_points
    from PIL import Image

    dataset = NuScenes(version="v1.0-mini", dataroot=str(root), verbose=False)
    scenes = sorted(dataset.scene, key=lambda scene: scene["token"])
    if len(scenes) != 10:
        raise ValueError("Expected 10 nuScenes-mini scenes")
    for index, scene in enumerate(scenes):
        token = scene["first_sample_token"]
        while token:
            sample = dataset.get("sample", token)
            lidar = dataset.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego = dataset.get("ego_pose", lidar["ego_pose_token"])["translation"]
            crops = {}
            for channel, camera_token in sample["data"].items():
                if not channel.startswith("CAM_"):
                    continue
                path, boxes, intrinsic = dataset.get_sample_data(camera_token, BoxVisibility.ANY)
                with Image.open(path) as image:
                    for box in boxes:
                        corners = box.corners()
                        visible = corners[:, corners[2] > 0.1]
                        if not visible.shape[1]:
                            continue
                        xy = view_points(visible, np.asarray(intrinsic), normalize=True)[:2]
                        low = np.maximum(np.floor(xy.min(axis=1)), 0).astype(int)
                        high = np.minimum(np.ceil(xy.max(axis=1)), image.size).astype(int)
                        if np.all(high > low):
                            crop = image.crop((*low, *high)).convert("RGB").resize((64, 64))
                            crops.setdefault(box.token, []).append(np.asarray(crop))
            for annotation in sample["anns"]:
                item = dataset.get("sample_annotation", annotation)
                category = item["category_name"]
                label = 0 if category.startswith("vehicle.") else 1 if category.startswith("human.pedestrian.") else 2
                distance = np.linalg.norm(np.asarray(item["translation"]) - ego)
                speed = np.linalg.norm(dataset.box_velocity(annotation))
                row = (scene["token"], partition(index), category, *item["size"], distance,
                       int(item["visibility_token"] or 0), item["num_lidar_pts"],
                       item["num_radar_pts"], float(np.nan_to_num(speed)))
                yield row, [(crop, label) for crop in crops.get(annotation, [])]
            token = sample["next"]


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--mock", action="store_true")
    modes.add_argument("--nuscenes-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images, labels = [], []
    with (args.output / "objetos.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(FIELDS)
        for row, crops in mock() if args.mock else nuscenes(args.nuscenes_dir):
            writer.writerow(row)
            for image, label in crops:
                images.append(image)
                labels.append((row[0], row[1], label))
    np.save(args.output / "recortes.npy", np.asarray(images, dtype=np.uint8).reshape(-1, 64, 64, 3))
    dtype = [("scene_token", "U32"), ("split", "U5"), ("label", "i8")]
    np.save(args.output / "recortes_labels.npy", np.asarray(labels, dtype=dtype))


if __name__ == "__main__":
    main()
