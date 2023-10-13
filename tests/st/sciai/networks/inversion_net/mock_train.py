"""mock train.py for inversion_net"""
import yaml
from sciai.utils import parse_arg
from sciai.context import init_project
from sciai.model.inversion_net.train import main


if __name__ == "__main__":
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    with open(f"./data/data_config.yaml") as f:
        data_config = yaml.safe_load(f)
    if config["amp_level"] == "O0":
        config["data_type"] = "float32"
    else:
        config["data_type"] = "float16"

    args = parse_arg(config)
    init_project(args=args)
    main(args, data_config)
