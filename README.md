# ABR wrapper for StanfordSNR/puffer 

## Adding a controller
To add a controller, create a class extending `BaseEnv`, implement the functions `setup_env(self, model_path: str) -> Callable` and `process_env_info(self, env_info: dict) -> Any` from `BaseEnv`, and add the class as one of the options in `test.py`. 

`process_env_info` is the function that will take the dictionary of the current client state `env_info`, and return the corresponding processed input to be given to the callable returned in `setup_env`. The callable returned by `setup_env` takes the current processed input of the environment state, and returns an int indicating which video format to send to the client next. `process_env_info` and `setup_env` will be called by the environment when necessary to make the abr controller work.

This means that `process_env_info` handles everything from frame stacking, invalid input by the Puffer and normalization, while `setup_env` simply returns a model that the env can query when needed. 

For an example, see `tara_env.py`.
