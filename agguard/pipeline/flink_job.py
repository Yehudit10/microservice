import os, json, boto3, cv2, numpy as np, yaml, logging

from pyflink.datastream import StreamExecutionEnvironment, RuntimeContext
from pyflink.datastream.functions import KeyedProcessFunction
from pyflink.datastream.connectors.kafka import (
    KafkaSource,
    KafkaSink,
    KafkaOffsetsInitializer,
    KafkaRecordSerializationSchema,   # ✅ add this
)
from pyflink.common import WatermarkStrategy, Types
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.connectors.kafka import DeliveryGuarantee



from agguard.pipeline.flink_manager import FlinkPipelineManager
from agguard.core.events.models import Rule

log = logging.getLogger("flink")



# def load_frame(bucket, key):
#     s3 = boto3.client("s3", endpoint_url="http://minio:9000")
#     obj = s3.get_object(Bucket=bucket.strip(), Key=key.strip())
#     data = obj["Body"].read()
#     arr = np.frombuffer(data, np.uint8)
#     return cv2.imdecode(arr, cv2.IMREAD_COLOR)

class CameraOperator(KeyedProcessFunction):
    def open(self, ctx: RuntimeContext):
        # Load the same config as gRPC server
        cfg_path = os.getenv("PIPELINE_CFG", "/app/configs/default.yaml")
        cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

        # ---- Setup logging ----
        log_level = cfg.get("logging", {}).get("level", "INFO")
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

        # ---- Create S3 client ----
        from agguard.adapters.s3_client import S3Client, S3Config
        s3_cfg = cfg.get("s3", {})
        self.s3 = S3Client(S3Config(
            region_name=s3_cfg.get("region_name", "us-east-1"),
            aws_access_key_id=s3_cfg.get("aws_access_key_id"),
            aws_secret_access_key=s3_cfg.get("aws_secret_access_key"),
            endpoint_url=s3_cfg.get("endpoint_url"),
            connect_timeout=float(s3_cfg.get("connect_timeout", 3.0)),
            read_timeout=float(s3_cfg.get("read_timeout", 10.0)),
            max_attempts=int(s3_cfg.get("max_attempts", 3)),
        ))

        # ---- Rules (same as in grpc_server) ----
        from agguard.core.events.models import Rule
        rules = [
            Rule(
                name="climbing_fence",
                target_cls="animal",
                target_cls_id=1,
                attr_value="object climbing a fence",
                severity=4,
                min_conf=0.5,
                min_consec=2,
                cooldown=12,
            ),
            # Add other rules here if needed
        ]

        # ---- Create pipeline manager ----
        from agguard.pipeline.flink_manager import FlinkPipelineManager
        self.pm = FlinkPipelineManager(cfg, self.s3, rules)

        log.info("[Flink] Initialized CameraOperator with %d rules and S3 at %s",
                len(rules), s3_cfg.get("endpoint_url"))
     

    def process_element(self, msg, ctx):
        data = json.loads(msg)
        # frame = load_frame(data["s3_bucket"], data["s3_key"])
        frame = self.s3.fetch_image_bgr(data["s3_bucket"], data["s3_key"])

        evt = self.pm.process(
            camera_id=data["camera_id"],
            ts_sec=float(data["ts_millis"]) / 1000.0,
            frame_idx=int(data.get("frame_idx", 0)),
            frame_bgr=frame
        )
        if evt:
            yield json.dumps(evt)

def main():
    bootstrap = os.getenv("KAFKA_BROKERS", "kafka:9092")
    topic_in = os.getenv("IN_TOPIC", "dev-camera-security")
    topic_out = os.getenv("OUT_TOPIC", "incidents.events")

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)

    source = (
        KafkaSource.builder()
        .set_bootstrap_servers(bootstrap)
        .set_topics(topic_in)
        .set_group_id("flink-camera-pipeline")
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )

    sink = (
    KafkaSink.builder()
    .set_bootstrap_servers(bootstrap)
    .set_record_serializer(
        KafkaRecordSerializationSchema.builder()
            .set_topic(topic_out)
            .set_value_serialization_schema(SimpleStringSchema())
            .build()
    )
    .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE)
    .build()
)

    stream = env.from_source(source, WatermarkStrategy.no_watermarks(), "CameraFrames")
    
    stream.key_by(lambda m: json.loads(m)["camera_id"]) \
    .process(CameraOperator(), output_type=Types.STRING()) \
    .sink_to(sink)

    env.execute("AgGuard Flink Pipeline")

if __name__ == "__main__":
    main()
