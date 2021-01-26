import configparser
import fire
import yaml


with open("/app/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def generate_config():
    """
    generate supervisord_<service>.conf files per service specified in config["sentinel"]["supervisord"]

    :return:
    """

    supervisor_config_parser = configparser.ConfigParser()

    for service in config["sentinel"]["supervisord"]:
        supervisor_config_parser.read_dict(config["sentinel"]["supervisord"][service])

        with open(f"/app/supervisord_{service}.conf", "w") as supervisor_config_file:
            supervisor_config_parser.write(supervisor_config_file)


if __name__ == "__main__":
    fire.Fire(generate_config)
