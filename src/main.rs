mod availability;
mod devices;

use availability::{get_gpu_availability, GPUAvailability};
use clap::Parser;
use colored::Colorize;
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

    // Number of min GPUs to use
    #[arg(long, default_value = "0")]
    min_gpus: usize,

    // Number of max GPUs to use
    #[arg(long, default_value = "1024")] // 1024 is a big enough number
    max_gpus: usize,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let mut devices = if let Some(selected_devices) = args.devices {
        let visible_devices = devices::get_visible_devices().unwrap();
        let selected_devices = devices::parse_cuda_visible_devices(&selected_devices).unwrap();
        devices::pick_devices(&selected_devices, &visible_devices)
    } else {
        devices::get_visible_devices().unwrap()
    };

    if devices.len() < args.min_gpus {
        println!(
            "{} You are trying to use {} GPU(s), but at least {} GPU(s) are required.",
            "ERROR:".bold().red(),
            devices.len(),
            args.min_gpus
        );
        return ExitCode::FAILURE;
    }

    if devices.len() > args.max_gpus {
        println!(
            "{} You are trying to use {} GPU(s), but at most {} GPU(s) are allowed.",
            "WARNING:".bold().yellow(),
            devices.len(),
            args.max_gpus
        );
        println!("Only the first {} GPU(s) will be used.", args.max_gpus);
        devices.truncate(args.max_gpus);
    }

    let devices = devices;

    return match get_gpu_availability(&devices, args.memory_border_mib) {
        Ok(GPUAvailability::Vacant) => {
            let devices_str = devices
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(",");

            println!(
                "{} GPU {} {} available!",
                "OK:".bold().green(),
                &devices_str,
                if devices.len() > 1 { "are" } else { "is" }
            );
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
                println!(
                    "{} {}",
                    "WARNING:".bold().yellow(),
                    "No command to execute.".bold()
                );
                println!("If you use this command with `&&`, use `--` instead.");
                println!(
                    "{}\n",
                    "Example: `knock-on-gpus --devices 0,1 -- python train.py`".dimmed()
                );
            }
            ExitCode::SUCCESS
        }
        Ok(GPUAvailability::Occupied(status)) => {
            println!(
                "{} GPU {} is currently in use",
                "ERROR:".bold().red(),
                status.id
            );
            println!("{:?}", status);
            println!("See `nvidia-smi` for more information.");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!(
                "{} Error has occurred while checking GPU usage:",
                "ERROR".bold().red()
            );
            eprintln!("{}", e);
            ExitCode::FAILURE
        }
    };
}
