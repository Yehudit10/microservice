from __future__ import annotations

import time
from datetime import timezone
from pathlib import Path
from typing import Iterable, List, Dict, Optional

import grpc
import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from agguard.proto import ingest_pb2, ingest_pb2_grpc

# =========================
# HARD-CODED CONFIG
# =========================
# gRPC target
GRPC_HOST = "security"
GRPC_PORT = 50051

# Camera / request behavior
CAMERA_ID = "dev-a"
START_FRAME_IDX = 1          # set to 0 to let server auto-increment
RETURN_DETECTIONS = True
DEADLINE_SECS = 30.0

# S3 listing settings (uses ambient AWS credentials or env vars if keys are None)
S3_BUCKET = "imagery"
S3_PREFIX = "dev_a"          # optional prefix inside the bucket, no leading slash

AWS_REGION = "us-east-1"
AWS_ACCESS_KEY_ID = "minioadmin"          # or None / "AKIA..."
AWS_SECRET_ACCESS_KEY = "minioadmin123"   # or None / "...secret..."
# For AWS S3 leave as None. For MinIO, e.g.: "http://localhost:9000"
S3_ENDPOINT_URL = "http://host.docker.internal:9001"

# File filtering (by key suffix)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Optional: limit how many keys to process (None = all)
MAX_KEYS: Optional[int] = None
# =========================


def _mk_boto3_s3():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        endpoint_url=S3_ENDPOINT_URL,
        config=BotoConfig(connect_timeout=3, read_timeout=10, retries={"max_attempts": 3}),
    )


def _is_image_key(key: str) -> bool:
    # Simple suffix filter; adjust if you store non-standard extensions.
    return any(key.lower().endswith(ext) for ext in IMAGE_EXTS)


def list_s3_images(bucket: str, prefix: str | None = None, max_keys: Optional[int] = None) -> List[Dict]:
    """
    List image objects in S3 under (bucket, prefix). Returns a list of records:
    { "bucket": str, "key": str, "ts_millis": int }
    - ts_millis is derived from the object's LastModified time (UTC) if available,
      otherwise current time.
    """
    s3 = _mk_boto3_s3()
    records: List[Dict] = []

    kwargs = {"Bucket": bucket}
    if prefix:
        kwargs["Prefix"] = prefix.strip().lstrip("/")

    continuation: Optional[str] = None
    try:
        while True:
            if continuation:
                kwargs["ContinuationToken"] = continuation

            resp = s3.list_objects_v2(**kwargs)

            contents = resp.get("Contents", [])
            for obj in contents:
                key = obj["Key"]
                if not _is_image_key(key):
                    continue

                lm = obj.get("LastModified")
                if lm is not None:
                    # Convert to epoch millis (ensure timezone-aware)
                    if lm.tzinfo is None:
                        lm = lm.replace(tzinfo=timezone.utc)
                    ts_millis = int(lm.timestamp() * 1000)
                else:
                    ts_millis = int(time.time() * 1000)

                records.append({"bucket": bucket, "key": key, "ts_millis": ts_millis})

                if max_keys is not None and len(records) >= max_keys:
                    break

            if max_keys is not None and len(records) >= max_keys:
                break

            if resp.get("IsTruncated"):
                continuation = resp.get("NextContinuationToken")
                if not continuation:
                    break
            else:
                break

    except (BotoCoreError, ClientError) as e:
        print(f"S3 list failed: {e}")

    # Keep stable order by key (lexicographic) to align with frame_idx increments
    records.sort(key=lambda r: r["key"])
    return records


def call_microservice_for_records(stub, records: list[dict]):
    """
    Call the gRPC microservice for each existing S3 object record.
    Record fields: bucket, key, ts_millis
    """
    frame_idx = START_FRAME_IDX if START_FRAME_IDX > 0 else 0

    for rec in records:
        req = ingest_pb2.ProcessImageRequest(
            s3_bucket=rec["bucket"],
            s3_key=rec["key"],
            camera_id=CAMERA_ID,
            ts_millis=rec["ts_millis"],
            frame_idx=(frame_idx if frame_idx > 0 else 0),  # 0 == unset for server
            return_detections=RETURN_DETECTIONS,
        )

        try:
            resp = stub.ProcessImage(req, timeout=DEADLINE_SECS)

            print(
                f"{rec['key']}: camera_id={resp.camera_id} "
                f"frame_idx={resp.frame_idx} change={resp.change_score:.3f} "
                f"dets={resp.num_detections} tracks={resp.num_tracks} "
                f"fps≈{resp.fps_estimate:.1f}"
            )

            if resp.opened_incident_id:
                print(f"  opened_incident_id:  {resp.opened_incident_id}")
            if resp.updated_incident_id:
                print(f"  updated_incident_id: {resp.updated_incident_id}")
            if resp.closed_incident_id:
                print(f"  closed_incident_id:  {resp.closed_incident_id}")

            if RETURN_DETECTIONS and len(resp.boxes):
                print("  boxes:")
                for b in resp.boxes:
                    print(
                        f"    ({b.x1},{b.y1})–({b.x2},{b.y2}) "
                        f"cls='{b.cls}' conf={b.conf:.2f} track_id={b.track_id}"
                    )

        except grpc.RpcError as e:
            print(f"RPC failed for {rec['key']}: {e.code().name} - {e.details()}")
            # Continue to next record
        finally:
            if frame_idx > 0:
                frame_idx += 1


def main():
    # Prepare gRPC client
    target = f"{GRPC_HOST}:{GRPC_PORT}"
    channel = grpc.insecure_channel(target)
    stub = ingest_pb2_grpc.ImageIngestorStub(channel)

    # List existing images from S3
    print(f"Listing images from s3://{S3_BUCKET}/{S3_PREFIX or ''} ...")
    records = list_s3_images(S3_BUCKET, S3_PREFIX, max_keys=MAX_KEYS)
    print(f"Found {len(records)} image object(s).")

    if not records:
        print("No images found in S3; aborting gRPC calls.")
        return

    print("Starting gRPC calls for listed S3 images...")
    call_microservice_for_records(stub, records)
    print("Done.")


if __name__ == "__main__":
    main()



# # tools/grpc_client.py
# from __future__ import annotations

# import time
# import mimetypes
# from pathlib import Path
# from concurrent.futures import ThreadPoolExecutor, as_completed

# import grpc
# import boto3
# from botocore.config import Config as BotoConfig
# from botocore.exceptions import BotoCoreError, ClientError

# from agguard.proto import ingest_pb2, ingest_pb2_grpc

# # =========================
# # HARD-CODED CONFIG
# # =========================
# # gRPC target
# GRPC_HOST = "security"
# GRPC_PORT = 50051

# # Camera / request behavior
# CAMERA_ID = "dev-a"
# START_FRAME_IDX = 1          # set to 0 to let server auto-increment
# RETURN_DETECTIONS = True
# DEADLINE_SECS = 30.0

# # Local images directory (searched recursively)
# # IMAGES_DIR = "D:\\output_frames"
# IMAGES_DIR = "/data/frames"

# # S3 settings (uses ambient AWS credentials or env vars if keys are None)
# S3_BUCKET = "imagery"
# S3_PREFIX = "dev_a"  # optional prefix inside the bucket, no leading slash

# AWS_REGION = "us-east-1"
# AWS_ACCESS_KEY_ID = "minioadmin"          # or "AKIA..."
# AWS_SECRET_ACCESS_KEY = "minioadmin123"   # or "...secret..."

# # For AWS S3 leave as None. For MinIO, e.g.: "http://localhost:9000"
# # S3_ENDPOINT_URL = "http://minio:9002"
# S3_ENDPOINT_URL = "http://host.docker.internal:9001"

# # Concurrency (only affects upload phase)
# UPLOAD_CONCURRENCY = 8

# # Fail fast if no images found
# FAIL_IF_EMPTY = True
# # =========================


# def _content_type_for(path: Path) -> str:
#     ctype, _ = mimetypes.guess_type(str(path))
#     return ctype or "application/octet-stream"


# def _mk_boto3_s3():
#     return boto3.client(
#         "s3",
#         region_name=AWS_REGION,
#         aws_access_key_id=AWS_ACCESS_KEY_ID,
#         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#         endpoint_url=S3_ENDPOINT_URL,
#         config=BotoConfig(connect_timeout=3, read_timeout=10, retries={"max_attempts": 3}),
#     )


# def _s3_key_for(path: Path) -> str:
#     """
#     Decide how to name the object in S3. Here we just use the local filename
#     under the configured prefix, which is simple and predictable.
#     """
#     prefix = (S3_PREFIX or "").strip().strip("/")
#     if prefix:
#         return f"{prefix}/{path.name}"
#     return path.name


# def iter_images(root: Path):
#     exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
#     for p in sorted(root.rglob("*")):
#         if p.is_file() and p.suffix.lower() in exts:
#             yield p


# def _upload_one(s3, local_path: Path) -> dict | None:
#     """
#     Upload a single image. Returns a record dict on success, or None on failure.
#     Record fields: bucket, key, ts_millis, local_path
#     """
#     key = _s3_key_for(local_path)
#     extra = {"ContentType": _content_type_for(local_path)}
#     try:
#         s3.upload_file(
#             Filename=str(local_path),
#             Bucket=S3_BUCKET,
#             Key=key,
#             ExtraArgs=extra,
#         )
#         # Use file mtime as capture timestamp (ms); fall back to now if needed.
#         try:
#             ts_millis = int(local_path.stat().st_mtime * 1000)
#         except Exception:
#             ts_millis = int(time.time() * 1000)

#         return {
#             "bucket": S3_BUCKET,
#             "key": key,
#             "ts_millis": ts_millis,
#             "local_path": local_path,
#         }
#     except (BotoCoreError, ClientError) as e:
#         print(f"S3 upload failed for {local_path}: {e}")
#         return None


# def upload_all_images(images: list[Path]) -> list[dict]:
#     """
#     Upload all images to S3 (concurrently). Returns a list of successful records.
#     """
#     s3 = _mk_boto3_s3()
#     if UPLOAD_CONCURRENCY <= 1:
#         records = []
#         for p in images:
#             rec = _upload_one(s3, p)
#             if rec:
#                 records.append(rec)
#         return records

#     records: list[dict] = []
#     with ThreadPoolExecutor(max_workers=UPLOAD_CONCURRENCY) as ex:
#         futures = {ex.submit(_upload_one, s3, p): p for p in images}
#         for fut in as_completed(futures):
#             rec = fut.result()
#             if rec:
#                 records.append(rec)
#     # Keep stable order by original filename to align with frame_idx increments
#     records.sort(key=lambda r: r["local_path"].name)
#     return records


# def call_microservice_for_records(stub, records: list[dict]):
#     """
#     After uploads complete, call the gRPC microservice for each uploaded object.
#     """
#     frame_idx = START_FRAME_IDX if START_FRAME_IDX > 0 else 0

#     for rec in records:
#         req = ingest_pb2.ProcessImageRequest(
#             s3_bucket=rec["bucket"],
#             s3_key=rec["key"],
#             camera_id=CAMERA_ID,
#             ts_millis=rec["ts_millis"],
#             frame_idx=(frame_idx if frame_idx > 0 else 0),  # 0 == unset for server
#             return_detections=RETURN_DETECTIONS,
#         )

#         try:
#             resp = stub.ProcessImage(req, timeout=DEADLINE_SECS)

#             print(
#                 f"{rec['local_path'].name}: camera_id={resp.camera_id} "
#                 f"frame_idx={resp.frame_idx} change={resp.change_score:.3f} "
#                 f"dets={resp.num_detections} tracks={resp.num_tracks} "
#                 f"fps≈{resp.fps_estimate:.1f}"
#             )

#             if resp.opened_incident_id:
#                 print(f"  opened_incident_id:  {resp.opened_incident_id}")
#             if resp.updated_incident_id:
#                 print(f"  updated_incident_id: {resp.updated_incident_id}")
#             if resp.closed_incident_id:
#                 print(f"  closed_incident_id:  {resp.closed_incident_id}")

#             if RETURN_DETECTIONS and len(resp.boxes):
#                 print("  boxes:")
#                 for b in resp.boxes:
#                     print(
#                         f"    ({b.x1},{b.y1})–({b.x2},{b.y2}) "
#                         f"cls='{b.cls}' conf={b.conf:.2f} track_id={b.track_id}"
#                     )

#         except grpc.RpcError as e:
#             print(
#                 f"RPC failed for {rec['local_path'].name}: "
#                 f"{e.code().name} - {e.details()}"
#             )
#             # Continue to next record
#         finally:
#             if frame_idx > 0:
#                 frame_idx += 1


# def main():
#     img_root = Path(IMAGES_DIR)
#     if not img_root.exists():
#         raise FileNotFoundError(f"Images directory not found: {img_root}")

#     images = list(iter_images(img_root))
#     if not images and FAIL_IF_EMPTY:
#         raise RuntimeError(f"No images found under {img_root}")

#     print(f"Found {len(images)} image(s). Uploading to s3://{S3_BUCKET}/{S3_PREFIX or ''} ...")
#     records = upload_all_images(images)
#     print(f"Uploaded {len(records)}/{len(images)} image(s).")

#     if not records:
#         print("No images uploaded successfully; aborting gRPC calls.")
#         return

#     # Prepare gRPC client
#     target = f"{GRPC_HOST}:{GRPC_PORT}"
#     channel = grpc.insecure_channel(target)
#     stub = ingest_pb2_grpc.ImageIngestorStub(channel)

#     print("Starting gRPC calls for uploaded images...")
#     call_microservice_for_records(stub, records)
#     print("Done.")


# if __name__ == "__main__":
#     main()

