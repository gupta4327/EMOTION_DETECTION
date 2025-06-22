import yaml
import sys
import os

yaml_file = sys.argv[1]
with open(yaml_file) as f:
    data = yaml.safe_load(f)
    with open(os.environ["GITHUB_ENV"], "a") as env_file:
        for k, v in data.items():
            clean_val = str(v).strip().strip('"').strip("'")   # removes surrounding quotes, just in case
            env_file.write(f'{k}="{clean_val}"\n')
