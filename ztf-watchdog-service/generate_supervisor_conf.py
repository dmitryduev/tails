import argparse
import sys
import yaml


def generate_conf(service):
    if service not in config["supervisord"].keys():
        raise ValueError(f"{service} service def not found in config['supervisord']")

    sc = ""
    for k in config["supervisord"][service].keys():
        sc += f"[{k}]\n"
        for kk, vv in config["supervisord"][service][k].items():
            sc += f"{kk} = {vv}\n"

    with open(f"/app/supervisord_{service}.conf", "w") as fsc:
        fsc.write(sc)


def watcher(args):
    generate_conf("watcher")


with open("/app/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="commands", dest="command")

    commands = [
        ("watcher", "generate supervisord_watcher.conf"),
        ("help", "Print this message"),
    ]

    parsers = {}
    for (cmd, desc) in commands:
        parsers[cmd] = subparsers.add_parser(cmd, help=desc)

    args = parser.parse_args()
    if args.command is None or args.command == "help":
        parser.print_help()
    else:
        getattr(sys.modules[__name__], args.command)(args)
