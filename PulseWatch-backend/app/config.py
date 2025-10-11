import os
import platform
import yaml

os_type=platform.system() 

config_path=os.path.join("user_config.yml")

with open(config_path, "r") as f:
    USER_CONFIG=yaml.safe_load(f)

prometheus_url = os.getenv("PROMETHEUS_URL") or USER_CONFIG["prometheus"]["url"]

metrics_config=USER_CONFIG["metrics"]

metric_queries = {
    "cpu": metrics_config["cpu"]["query"][os_type],
    "memory": metrics_config["memory"]["query"][os_type],
    "latency": metrics_config["latency"]["query"][os_type],  
}

model_config = USER_CONFIG.get("model", {})
