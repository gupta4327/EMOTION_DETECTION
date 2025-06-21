import yaml

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except FileNotFoundError:
        raise
    except yaml.YAMLError as e:
        raise
    except Exception as e:
        raise
