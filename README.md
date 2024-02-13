# knock-on-gpus

A CLI tool for checking if GPUs are available before running your script that uses GPUs.

## Usage

### Basic usage

You can use `knock-on-gpus` to run a script that uses GPUs.

```bash
knock-on-gpus && python my_script.py
```

If some GPUs are not available, `knock-on-gpus` will return an error code and print a message to the console.

### Using with `CUDA_VISIBLE_DEVICES`

You can also use `knock-on-gpus` to run a script with specific GPUs.

```bash
CUDA_VISIBLE_DEVICES=0 knock-on-gpus && python my_script.py
```

### Set alias for `python`

You can set an alias for `python` to use `knock-on-gpus` by default.

```bash
alias python='knock-on-gpus && python'
```

Then you can run your script without `knock-on-gpus`.

```bash

## Options

### `--silent`

Suppresses the error message when GPUs are not available.

### `--verbose`

Prints success messages to the console.

### `--all`

Checks if all GPUs are available even if `CUDA_VISIBLE_DEVICES` is set.

## Installation

Under construction.
