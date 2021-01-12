#!/usr/bin/env python
from contextlib import contextmanager
from deepdiff import DeepDiff
from distutils.version import LooseVersion as Version
import pathlib
from pprint import pprint
import questionary
import subprocess
import sys
import time
import typer

# from typing import Optional
import yaml


def output(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = p.communicate()
    success = p.returncode == 0
    return success, out


deps = {
    "python": (
        # Command to get version
        ["python", "--version"],
        # Extract *only* the version number
        lambda v: v.split()[1],
        # It must be >= 3.7
        "3.7",
    ),
    "docker": (
        # Command to get version
        ["docker", "--version"],
        # Extract *only* the version number
        lambda v: v.split()[2][:-1],
        # It must be >= 18.06
        "18.06",
    ),
    "docker-compose": (
        # Command to get version
        ["docker-compose", "--version"],
        # Extract *only* the version number
        lambda v: v.split()[2][:-1],
        # It must be >= 1.22.0
        "1.22.0",
    ),
}


@contextmanager
def status(message):
    """
    Borrowed from https://github.com/cesium-ml/baselayer/
    :param message:
    :return:
    """
    print(f"[·] {message}", end="")
    sys.stdout.flush()
    try:
        yield
    except Exception:
        print(f"\r[✗] {message}")
        raise
    else:
        print(f"\r[✓] {message}")


def deps_ok():
    """
    Borrowed from https://github.com/fritz-marshal/fritz
    :return:
    """
    print("Checking system dependencies:")

    fail = []

    for dep, (cmd, get_version, min_version) in deps.items():
        try:
            query = f"{dep} >= {min_version}"
            with status(query):
                success, out = output(cmd)
                try:
                    version = get_version(out.decode("utf-8").strip())
                    print(f"[{version.rjust(8)}]".rjust(40 - len(query)), end="")
                except Exception:
                    raise ValueError("Could not parse version")

                if not (Version(version) >= Version(min_version)):
                    raise RuntimeError(f"Required {min_version}, found {version}")
        except ValueError:
            print(
                f"\n[!] Sorry, but our script could not parse the output of "
                f'`{" ".join(cmd)}`; please file a bug, or see '
                f"`check_app_environment.py`\n"
            )
            raise
        except Exception as e:
            fail.append((dep, e))

    if fail:
        print()
        print("[!] Some system dependencies seem to be unsatisfied")
        print()
        print("    The failed checks were:")
        print()
        for (pkg, exc) in fail:
            cmd, get_version, min_version = deps[pkg]
            print(f'    - {pkg}: `{" ".join(cmd)}`')
            print("     ", exc)
        print()
        print(
            "    Please refer to https://github.com/dmitryduev/tails "
            "for installation instructions."
        )
        print()
        return False

    print("-" * 20)
    return True


def check_configs(config_wildcards=("config.*yaml", "docker-compose.*yaml")):
    path = pathlib.Path(__file__).parent.absolute()

    for config_wildcard in config_wildcards:
        config = config_wildcard.replace("*", "")
        # use config defaults if configs do not exist?
        if not (path / config).exists():
            answer = questionary.select(
                f"{config} does not exist, do you want to use one of the following"
                " (not recommended without inspection)?",
                choices=[p.name for p in path.glob(config_wildcard)],
            ).ask()
            subprocess.run(["cp", f"{path / answer}", f"{path / config}"])

        # check contents of config.yaml WRT config.defaults.yaml
        if config == "config.yaml":
            with open(path / config.replace(".yaml", ".defaults.yaml")) as config_yaml:
                config_defaults = yaml.load(config_yaml, Loader=yaml.FullLoader)
            with open(path / config) as config_yaml:
                config_wildcard = yaml.load(config_yaml, Loader=yaml.FullLoader)
            deep_diff = DeepDiff(config_wildcard, config_defaults, ignore_order=True)
            difference = {
                k: v
                for k, v in deep_diff.items()
                if k in ("dictionary_item_added", "dictionary_item_removed")
            }
            if len(difference) > 0:
                print("config.yaml structure differs from config.defaults.yaml")
                pprint(difference)
                raise KeyError("Fix config.yaml before proceeding")


app = typer.Typer()


@app.command()
def fetch_models():
    """Fetch Tails models from GCP"""
    path_models = pathlib.Path(__file__).parent / "models"
    if not path_models.exists():
        path_models.mkdir(parents=True, exist_ok=True)

    command = [
        "gsutil",
        "-m",
        "cp",
        "-r",
        "gs://tails-models/*",
        str(path_models),
    ]
    p = subprocess.run(command, check=True)
    if p.returncode != 0:
        raise RuntimeError("Failed to fetch Tails models")


@app.command()
def build():
    """Build containers"""
    check_configs()

    p = subprocess.run(
        ["docker-compose", "-f", "docker-compose.yaml", "build"],
        check=True,
    )
    if p.returncode != 0:
        raise RuntimeError("Failed to start watchdog")


@app.command()
def up(build: bool = False):
    """Start service"""
    # run service
    command = ["docker-compose", "-f", "docker-compose.yaml", "up", "-d"]
    if build:
        command.append("--build")
    p = subprocess.run(command, check=True)
    if p.returncode != 0:
        raise RuntimeError("Failed to start service")


@app.command()
def down():
    # shutdown service
    print("Shutting down service")
    command = ["docker-compose", "-f", "docker-compose.yaml", "down"]
    subprocess.run(command)


@app.command()
def test():
    """Test service"""
    print("Running the test suite")

    # make sure the containers are up and running
    num_retries = 10
    for i in range(num_retries):
        if i == num_retries - 1:
            raise RuntimeError("Watchdog's containers failed to spin up")

        command = ["docker", "ps", "-a"]
        container_list = (
            subprocess.check_output(command, universal_newlines=True)
            .strip()
            .split("\n")
        )
        if len(container_list) == 1:
            print("No containers are running, waiting...")
            time.sleep(2)
            continue

        containers_up = (
            len(
                [
                    container
                    for container in container_list
                    if container_name in container and " Up " in container
                ]
            )
            > 0
            for container_name in ("tails_watchdog_1",)
        )

        if not all(containers_up):
            print("Watchdog's containers are not up, waiting...")
            time.sleep(2)
            continue

        break

    print("Testing watchdog")

    command = [
        "docker",
        "exec",
        "-i",
        "tails_watcher_1",
        "python",
        "-m",
        "pytest",
        "-s",
        "test_watchdog.py",
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        sys.exit(1)


if __name__ == "__main__":
    # check environment
    env_ok = deps_ok()
    if not env_ok:
        raise RuntimeError("Halting because of unsatisfied system dependencies")

    app()
