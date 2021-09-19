# ABR wrapper for StanfordSNR/puffer 

## Adding a controller
To add a controller, create a class extending `BaseEnv`, implement the functions `setup_env(self, model_path: str) -> Callable` and `process_env_info(self, env_info: dict) -> Any` from `BaseEnv`, and add the class as one of the options in `test.py`. 

`process_env_info` is the function that will take the dictionary of the current client state `env_info`, and return the corresponding processed input to be given to the callable returned in `setup_env`. The callable returned by `setup_env` takes the current processed input of the environment state, and returns an int indicating which video format to send to the client next. `process_env_info` and `setup_env` will be called by the environment when necessary to make the abr controller work.

`process_env_info` will be passed in `env_info`, a dictionary with the following information:
```json
{
  "past_chunk": {
    "delay": "total time for server to ack chunk (s)",
    "ssim": "ssim (unitless)",
    "size": "size of chunk (Mb)",
    "cwnd": "TCP cwnd (packets)",
    "in_flight": "TCP packets sent not acked (packets)",
    "min_rtt": "TCP min rtt (s)",
    "rtt": "TCP rtt (s)",
    "delivery_rate": "TCP delivery rate (Mb/s)"
  },
  "buffer": "client buffer (s)",
  "cum_rebuf": "total time client spent rebuffering (s)",
  "sizes": "sizes of next 5 chunks (list[5,10], Mb)",
  "ssims": "ssims of next 5 chunks (list[5,10], unitless)",
  "channel_name": "name of channel currently being watched (string)"
}
```

For an example, see `tara_env.py`.
