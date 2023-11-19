import yaml
from yamlinclude import YamlIncludeConstructor

YamlIncludeConstructor.add_to_loader_class(
    loader_class=yaml.FullLoader, base_dir="./configs"
)


def get_cfg(yaml_path):
    with open(yaml_path, "r") as f:
        try:
            return yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)