import os
import sys
import json
import argparse
from dotenv import load_dotenv
from cvat_sdk import Client
from cvat_sdk.core.client import PasswordCredentials
from cvat_sdk.models import (
    LabeledDataRequest,
    LabeledShapeRequest,
    ShapeType,
)

# Load environment variables from .env if present
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Upload predicted segmentation masks to CVAT")
    parser.add_argument("--job-id", type=int, required=True, help="CVAT Job ID")
    parser.add_argument("--url", type=str, default=os.getenv("CVAT_URL"), help="CVAT server URL (or via CVAT_URL env var)")
    parser.add_argument("--username", type=str, default=os.getenv("CVAT_USERNAME"), help="CVAT Username (or via CVAT_USERNAME env var)")
    parser.add_argument("--password", type=str, default=os.getenv("CVAT_PASSWORD"), help="CVAT Password (or via CVAT_PASSWORD env var)")
    parser.add_argument("--predictions", type=str, default="predictions.json", help="Path to predictions JSON")
    parser.add_argument("--manual-frames", type=str, default="1,5,6,7", help="Comma-separated frame IDs to preserve (ground-truth)")
    return parser.parse_args()

def main():
    args = parse_args()

    if not args.url or not args.username or not args.password:
        print("ERROR: CVAT credentials must be provided via CLI flags (--url, --username, --password) or in a .env file.")
        sys.exit(1)

    if not os.path.exists(args.predictions):
        raise FileNotFoundError(f"Predictions file not found: {args.predictions}")

    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predicted frames from {args.predictions}")

    manual_frames = set(int(f.strip()) for f in args.manual_frames.split(",") if f.strip().isdigit())

    print(f"Connecting to CVAT at {args.url} as '{args.username}'...")
    credentials = PasswordCredentials(args.username, args.password)
    with Client(args.url, check_server_version=False) as client:
        client.login(credentials)
        print("Successfully authenticated to CVAT!")

        job = client.jobs.retrieve(args.job_id)
        labels = job.get_labels()
        redness_label = None
        for lbl in labels:
            if "redness" in lbl.name.lower() or "rash" in lbl.name.lower():
                redness_label = lbl
                break

        if redness_label is None:
            raise ValueError(f"Label 'redness_rash' not found among job labels: {[l.name for l in labels]}")

        print(f"Target label: '{redness_label.name}' (ID: {redness_label.id})")

        # Preserve manual annotations and non-redness labels
        existing_anns = job.get_annotations()
        preserved_shapes = []
        for s in existing_anns.shapes:
            if s.label_id != redness_label.id or s.frame in manual_frames:
                req = LabeledShapeRequest(
                    type=s.type,
                    frame=s.frame,
                    label_id=s.label_id,
                    points=s.points,
                    rotation=s.rotation,
                    z_order=s.z_order,
                    attributes=s.attributes,
                    occluded=s.occluded,
                    source=s.source,
                )
                preserved_shapes.append(req)

        print(f"Preserving {len(preserved_shapes)} existing shapes (manual frames {sorted(list(manual_frames))} + other classes).")

        new_shapes = []
        for item in predictions:
            frame_id = item["frame"]
            if frame_id in manual_frames:
                continue

            points = item.get("points")
            if not points or len(points) < 5:
                continue

            req = LabeledShapeRequest(
                type=ShapeType("mask"),
                frame=frame_id,
                label_id=redness_label.id,
                points=points,
                rotation=0.0,
                z_order=0,
                occluded=False,
                source="manual",
                attributes=[]
            )
            new_shapes.append(req)

        total_shapes = preserved_shapes + new_shapes
        print(f"\nUploading {len(total_shapes)} shapes to CVAT ({len(new_shapes)} updated masks + {len(preserved_shapes)} preserved)...")
        data_request = LabeledDataRequest(shapes=total_shapes)
        job.import_annotations(data_request)
        print("Upload successful!")

    print("\n" + "="*60)
    print("SUCCESS! CVAT ANNOTATIONS UPDATED FOR ALL FRAMES!")
    print(f"Job ID: {args.job_id}")
    print(f"Total new masks: {len(new_shapes)}")
    print(f"Preserved manual frames: {sorted(list(manual_frames))}")
    print("="*60)

if __name__ == "__main__":
    main()
