from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2, numpy as np
import botocore
import boto3
from botocore.config import Config as BotoConfig

@dataclass(frozen=True)
class S3Config:
    # For AWS S3 you usually set only region_name; credentials come from env/role.
    region_name: str = "us-east-1"
    # Optional explicit creds (otherwise rely on IAM role, env vars, or shared config)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None

    # Optional: only if you still want to talk to a non-AWS S3-compatible endpoint (e.g., MinIO)
    endpoint_url: Optional[str] = None
    # Optional knobs
    connect_timeout: float = 3.0
    read_timeout: float = 10.0
    max_attempts: int = 3     # boto retries on transient errors


class S3Client:
    def __init__(self, cfg: S3Config):
        self.cfg = cfg
        boto_cfg = BotoConfig(
            region_name=cfg.region_name,
            s3={"addressing_style": "path"},  # AWS default
            retries={"max_attempts": cfg.max_attempts, "mode": "standard"},
            connect_timeout=cfg.connect_timeout,
            read_timeout=cfg.read_timeout,
            # s3={"addressing_style": "path"},
        )
        session = boto3.Session(
            aws_access_key_id=cfg.aws_access_key_id,
            aws_secret_access_key=cfg.aws_secret_access_key,
            aws_session_token=cfg.aws_session_token,
            region_name=cfg.region_name,
        )
        self.s3 = session.client("s3", endpoint_url=cfg.endpoint_url, config=boto_cfg)

    def fetch_image_bgr(self, bucket: str, object_key: str) -> np.ndarray:
        """GET s3://bucket/object_key and decode as BGR numpy image."""
        try:
            resp = self.s3.get_object(Bucket=bucket, Key=object_key)
            data: bytes = resp["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            raise FileNotFoundError(f"s3://{bucket}/{object_key} not found (NoSuchKey)")
        except self.s3.exceptions.NoSuchBucket:
            raise FileNotFoundError(f"s3://{bucket} does not exist (NoSuchBucket)")
        except botocore.exceptions.EndpointConnectionError as e:
            raise RuntimeError(f"S3 endpoint unreachable: {e}")
        except botocore.exceptions.ClientError as e:
            # surface 403/404/etc with context
            code = e.response.get("Error", {}).get("Code")
            msg = e.response.get("Error", {}).get("Message")
            raise RuntimeError(f"S3 get_object failed ({code}): {msg}") from e

        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image bytes from s3://{bucket}/{object_key}")
        return img
    # add inside class S3Client:

    def put_file(self, bucket: str, key: str, local_path: str, content_type: Optional[str] = None) -> None:
        extra = {"ContentType": content_type} if content_type else {}
        self.s3.upload_file(local_path, bucket, key, ExtraArgs=extra or None)

    def delete_object(self, bucket: str, key: str) -> None:
        self.s3.delete_object(Bucket=bucket, Key=key)

    def delete_prefix(self, bucket: str, prefix: str) -> None:
        # batch delete up to 1000 keys per call
        token = None
        while True:
            resp = self.s3.list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=token) \
                   if token else self.s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            contents = resp.get("Contents", [])
            if contents:
                objects = [{"Key": obj["Key"]} for obj in contents]
                self.s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
    
    
    def get_object_stream(self, bucket: str, key: str, range_header: Optional[str] = None):
        """Return boto3 get_object response (Body is a stream)."""
        params = {"Bucket": bucket, "Key": key}
        print(params)
        if range_header:
            params["Range"] = range_header
        return self.s3.get_object(**params)
    
    # in agguard/adapters/s3_client.py
    def put_bytes(self, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream"):
        import io
        bio = io.BytesIO(data)
        self.client.upload_fileobj(bio, bucket, key, ExtraArgs={"ContentType": content_type})



