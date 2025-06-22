# scripts/load_env_from_yaml.py
import yaml
import sys
import os

yaml_file = sys.argv[1]  # e.g., "vars.yml"
with open(yaml_file) as f:
    data = yaml.safe_load(f)
    with open(os.environ["GITHUB_ENV"], "a") as env_file:
        for k, v in data.items():
            line = f'{k}="{v}"' if isinstance(v, str) else f'{k}="{str(v)}"'
            env_file.write(line + "\n")