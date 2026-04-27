import boto3
import time
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# -----------------------------
# CONFIG (fill these)
# -----------------------------
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
REGION = os.getenv("REGION")

CLUSTER = "transformer-cluster"
TASK_DEF = "transformer-taskdefinition"
CONTAINER_NAME = "transformer"  # must match task definition

SUBNETS = [
    "subnet-0e1868337d712692a",
    "subnet-017e670daf03c5dc7",
    "subnet-00134de6ad95062e2"
]

SECURITY_GROUPS = ["sg-06c75adc023f7d037"]

LOG_GROUP = "/ecs/transformer-taskdefinition"


# -----------------------------
# INPUT
# -----------------------------
START = "Shubham"
MAX_NEW_TOKENS = "100"


# -----------------------------
# SESSION
# -----------------------------
session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
ecs = session.client("ecs", region_name=REGION)
logs = session.client("logs", region_name=REGION)


# -----------------------------
# RUN TASK
# -----------------------------
response = ecs.run_task(
    cluster=CLUSTER,
    launchType="FARGATE",
    taskDefinition=TASK_DEF,
    count=1,
    overrides={
        "containerOverrides": [
            {
                "name": CONTAINER_NAME,
                "environment": [
                    {"name": "START", "value": START},
                    {"name": "MAX_NEW_TOKENS", "value": MAX_NEW_TOKENS}
                ]
            }
        ]
    },
    networkConfiguration={
        "awsvpcConfiguration": {
            "subnets": SUBNETS,
            "securityGroups": SECURITY_GROUPS,
            "assignPublicIp": "ENABLED"
        }
    }
)

tasks = response.get("tasks", [])
if not tasks:
    print("❌ Failed to start task:", response)
    exit()

task_arn = tasks[0]["taskArn"]
print("🚀 Started Task:", task_arn)


# -----------------------------
# WAIT UNTIL STOPPED
# -----------------------------
waiter = ecs.get_waiter("tasks_stopped")
waiter.wait(cluster=CLUSTER, tasks=[task_arn])
print("✅ Task finished")


# -----------------------------
# GET LOG STREAM
# -----------------------------
desc = ecs.describe_tasks(cluster=CLUSTER, tasks=[task_arn])
container = desc["tasks"][0]["containers"][0]

log_stream = container.get("logStreamName")

if not log_stream:
    print("❌ No logs found")
    exit()


# -----------------------------
# FETCH LOGS (FIXED)
# -----------------------------
time.sleep(5)

streams = logs.describe_log_streams(
    logGroupName=LOG_GROUP,
    orderBy="LastEventTime",
    descending=True,
    limit=1
)

if not streams["logStreams"]:
    print("❌ No log streams found")
    exit()

log_stream = streams["logStreams"][0]["logStreamName"]

events = logs.get_log_events(
    logGroupName=LOG_GROUP,
    logStreamName=log_stream,
    startFromHead=True
)

print("\n===== MODEL OUTPUT =====")
for e in events["events"]:
    print(e["message"])