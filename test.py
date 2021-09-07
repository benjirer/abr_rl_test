#!/usr/bin/env python3

# A default shebang is set, but it may not point to the expected python path
import sys
from deep_rl_env import DeepRlEnv


NAME_TO_ENV = {
    "deep_rl": DeepRlEnv
}

def main(args):
    env_cls = NAME_TO_ENV[args["name"]]
    env = env_cls(model_path=args["model_path"], 
        server_address=args["server_address"])
    env.env_loop()

if __name__ == "__main__":
    args = {"name": sys.argv[1], "model_path": sys.argv[2], 
                                        "server_address": sys.argv[3]}
    main(args)