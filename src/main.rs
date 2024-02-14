mod availability;
mod devices;
mod logger;

use availability::{get_gpu_availability, GPUAvailability};
use clap::Parser;
use colored::Colorize;
use log::{error, info, warn};
use std::process::ExitCode;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Set visible devices
    #[arg(short, long)]
    devices: Option<String>,

    #[arg(last = true)]
    commands: Vec<String>,

    #[arg(long)]
    memory_border_mib: Option<f32>,

    #[arg(long, default_value = "false")]
    use_gpu_strictly: bool,

    // Number of min GPUs to use
    #[arg(long, default_value = "0")]
    min_gpus: usize,

    // Number of max GPUs to use
    #[arg(long, default_value = "1024")] // 1024 is a big enough number
    max_gpus: usize,
}

fn main() -> ExitCode {
    logger::init_logger();
    let args = Args::parse();

    let is_cuda_available = devices::is_cuda_available();

    if !is_cuda_available && args.use_gpu_strictly {
        error!("CUDA is not available, but you are trying to use GPU strictly.");
        return ExitCode::FAILURE;
    }

    let mut devices = if !is_cuda_available {
        vec![]
    } else if let Some(selected_devices) = args.devices {
        let visible_devices = devices::get_visible_devices().unwrap();
        let selected_devices = devices::parse_cuda_visible_devices(&selected_devices).unwrap();
        devices::pick_devices(&selected_devices, &visible_devices).unwrap()
    } else {
        devices::get_visible_devices().unwrap()
    };

    if devices.len() < args.min_gpus {
        error!(
            "You are trying to use {} GPU(s), but at least {} GPU(s) are required.",
            devices.len(),
            args.min_gpus
        );
        return ExitCode::FAILURE;
    }

    if devices.len() > args.max_gpus {
        warn!(
            "You are trying to use {} GPU(s), but at most {} GPU(s) are allowed.\n\
            Only the first {} GPU(s) will be used.",
            devices.len(),
            args.max_gpus,
            args.max_gpus
        );
        devices.truncate(args.max_gpus);
    }

    let devices = devices;
    let availability = if devices.len() == 0 {
        Ok(GPUAvailability::Vacant)
    } else {
        get_gpu_availability(&devices, args.memory_border_mib)
    };

    match availability {
        Ok(GPUAvailability::Vacant) => {
            let devices_str = devices
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(",");

            if devices.len() == 0 {
                warn!("CUDA is not available, using CPU instead.");
            } else {
                info!(
                    "GPU {} {} available!",
                    &devices_str,
                    if devices.len() > 1 { "are" } else { "is" }
                );
            }
            if args.commands.len() > 0 {
                println!(); // Add a new line

                // Execute the command
                let mut status = std::process::Command::new(&args.commands[0])
                    .args(&args.commands[1..])
                    .envs(std::env::vars())
                    .env("CUDA_VISIBLE_DEVICES", devices_str)
                    .spawn()
                    .expect("Failed to execute command");

                status.wait().expect("Failed to wait for command");
            } else {
                warn!(
                    "No command to execute.\n\
                If you use this command with `&&`, use `--` instead.\n{}",
                    "Example: `knock-on-gpus --devices 0,1 -- python train.py`".dimmed()
                );
            }
            ExitCode::SUCCESS
        }

        Ok(GPUAvailability::Occupied(status)) => {
            error!(
                "GPU {} is currently in use\n{:?}\nSee `nvidia-smi` for more information.",
                status.id, status
            );
            ExitCode::FAILURE
        }

        Err(e) => {
            error!(
                "Error has occurred while checking GPU usage:\n{}",
                e.to_string().dimmed()
            );
            ExitCode::FAILURE
        }
    }
}
