# knock-on-gpus

A CLI tool for checking if GPUs are available before running your script that uses GPUs.

## Installation

```sh
pip install knock-on-gpus
```

## Quick start

### Basic usage

You can use `knock-on-gpus` to run a script that uses GPUs.

```bash
knock-on-gpus -- python my_script.py
```

If some GPUs are not available, `knock-on-gpus` will return an error code and print a message to the console.

### Using with `CUDA_VISIBLE_DEVICES`

You can also use `knock-on-gpus` to run a script with specific GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1 knock-on-gpus -- python my_script.py
```

You can also use `--devices` or `-d` to specify the GPUs to use.

```bash
knock-on-gpus -d 0,1 -- python my_script.py
```

### Auto selection

You can use `--auto-select` to automatically allocate the number of GPUs.

```bash
knock-on-gpus --auto-select 2 -- python my_script.py
```

If GPU:0, GPU:1, and GPU:3 are unavailable, `knock-on-gpus` will use GPU:2 and GPU:4.

### Set alias for `python`

You can set an alias for `python` to use `knock-on-gpus` by default.

```bash
alias unsafe-python="`which python`"
alias python="knock-on-gpus -- python"
```

Then you can run your script without `knock-on-gpus`.

## Options

### `--devices`

(Alias: `-d`, `--device`)

Specifies the GPUs to use. The value is a comma-separated list of GPU IDs.

### `--memory-border-mib`

Specifies the memory border (MiB) to treat as vacant. If the memory usage exceeds this value, the GPU will be treated as occupied.

### `--use-gpu-strictly`

If true, use GPU strictly. If CUDA is not available, it will fail.

### `--min-gpus`

Specifies the number of min GPUs to use.

### `--max-gpus`

Specifies the number of max GPUs to use.

### `--cuda-visible-devices-env-key`

Specifies the environment variable key to set visible devices.

### `--verbose`

If true, print verbose logs.

### `--auto-select`

(Alias: `-a`, `--auto`)

If a number is given, it will automatically allocate the number of GPUs.
