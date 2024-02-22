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
    /// Visible devices
    #[arg(short, long)]
    devices: Option<String>,

    /// Commands to execute after checking GPU availability.
    /// CUDA_VISIBLE_DEVICES will be set to the available devices.
    #[arg(last = true)]
    commands: Vec<String>,

    /// Memory border (MiB) to treat as vacant.
    /// If the memory usage exceeds this value, the GPU will be treated as occupied.
    #[arg(long, default_value = "300")]
    memory_border_mib: f32,

    /// If true, use GPU strictly. If CUDA is not available, it will fail.
    #[arg(long, default_value = "false")]
    use_gpu_strictly: bool,

    /// Number of min GPUs to use
    #[arg(long, default_value = "0")]
    min_gpus: usize,

    /// Number of max GPUs to use
    #[arg(long, default_value = "1024")] // 1024 is a big enough number
    max_gpus: usize,

    /// If a number is given, it will automatically allocate the number of GPUs.
    #[arg(long)]
    auto_select: Option<usize>,

    /// If true, print debug logs
    #[arg(long)]
    verbose: bool,

    /// Environment variable key to set visible devices
    #[arg(long, default_value = "CUDA_VISIBLE_DEVICES")]
    cuda_visible_devices_env_key: String,
}

fn main() -> ExitCode {
    let args = Args::parse();
    logger::init_logger(if args.verbose {
        log::Level::Debug
    } else {
        log::Level::Info
    });

    let is_cuda_available = devices::is_cuda_available();

    if !is_cuda_available && args.use_gpu_strictly {
        error!("CUDA is not available, but you are trying to use GPU strictly.");
        return ExitCode::FAILURE;
    }

    // Get visible devices using `--devices` and `CUDA_VISIBLE_DEVICES`
    let mut devices = if !is_cuda_available {
        vec![]
    } else if let Some(selected_devices) = args.devices {
        let key = &args.cuda_visible_devices_env_key;
        let visible_devices = devices::get_visible_devices(Some(key)).unwrap();
        let selected_devices = devices::parse_cuda_visible_devices(&selected_devices).unwrap();
        devices::pick_devices(&selected_devices, &visible_devices).unwrap()
    } else {
        let key = &args.cuda_visible_devices_env_key;
        devices::get_visible_devices(Some(key)).unwrap()
    };

    // Check if the number of devices is enough
    if devices.len() < args.min_gpus {
        error!(
            "You are trying to use {} GPU(s), but at least {} GPU(s) are required.",
            devices.len(),
            args.min_gpus
        );
        return ExitCode::FAILURE;
    }

    // Check if the number of devices is too much
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

    // Make devices immutable
    let devices = devices;

    // Check if selected devices are available
    let availability =
        get_gpu_availability(&devices, args.memory_border_mib, args.auto_select.clone());

    match availability {
        Ok(GPUAvailability::Vacant(devices)) => {
            let devices_str = devices
                .iter()
                .map(|x| x.id.to_string())
                .collect::<Vec<String>>()
                .join(",");

            if args.auto_select.as_ref().is_some() {
                info!("{} GPU(s) will be used.", devices.len());
            }

            // Print the message
            if devices.len() == 0 {
                warn!("CUDA is not available, using CPU instead.");
            } else {
                info!(
                    "GPU {} {} available!",
                    &devices_str,
                    if devices.len() > 1 { "are" } else { "is" }
                );
            }

            // If all devices are available, execute the command
            if args.commands.len() > 0 {
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
                    concat!(
                        "No command to execute.\n",
                        "If you use this command with `&&`, use `--` instead.\n",
                        "{}"
                    ),
                    "Example: `knock-on-gpus --devices 0,1 -- python train.py`".dimmed()
                );
            }
            ExitCode::SUCCESS
        }

        Ok(GPUAvailability::Occupied(status)) => {
            let devices_you_want = devices
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(",");
            let used_devices = status
                .iter()
                .filter(|s| !s.is_vacant)
                .map(|s| s.id.to_string())
                .collect::<Vec<String>>();
            error!(
                "You tried to use GPU {}, but GPU {} {} currently in use.\nSee `nvidia-smi` for more information.",
                &devices_you_want,
                &used_devices.join(","),
                if used_devices.len() > 1 { "are" } else { "is" }
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
