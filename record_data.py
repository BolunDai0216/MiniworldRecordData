#!/usr/bin/env python3

"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""

import argparse

import gymnasium as gym

import miniworld
from record_data_env import RecordDataEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="MiniWorld-OneRoom-v0")
    parser.add_argument(
        "--domain-rand", action="store_true", help="enable domain randomization"
    )
    parser.add_argument(
        "--no-time-limit", action="store_true", help="ignore time step limits"
    )
    parser.add_argument(
        "--top_view",
        action="store_true",
        help="show the top view instead of the agent view",
    )
    parser.add_argument("--filename", default="data/data.pickle")
    args = parser.parse_args()
    view_mode = "top" if args.top_view else "agent"

    env = gym.make(args.env_name, view=view_mode, render_mode="human")
    miniworld_version = miniworld.__version__

    print(f"Miniworld v{miniworld_version}, Env: {args.env_name}")

    record_data = RecordDataEnv(
        env, args.no_time_limit, args.domain_rand, args.filename
    )
    record_data.run()


if __name__ == "__main__":
    main()
