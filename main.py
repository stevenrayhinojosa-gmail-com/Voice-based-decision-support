from app import app  # noqa: F401
import json
import logging

# Load version information
try:
    with open('model_version.json', 'r') as f:
        version_info = json.load(f)
    logging.info(f"Running {version_info['model_name']} Model Version {version_info['version']}")
except Exception as e:
    logging.warning(f"Could not load version information: {e}")
    version_info = {"model_name": "SereniTeach", "version": "2.0"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
