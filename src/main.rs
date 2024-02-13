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
    last: Vec<String>,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let devices = if let Some(selected_devices) = args.devices {
        let visible_devices = devices::get_visible_devices().unwrap();
        let selected_devices = devices::parse_cuda_visible_devices(&selected_devices).unwrap();
        devices::pick_devices(&selected_devices, &visible_devices)
    } else {
        devices::get_visible_devices().unwrap()
    };

    return match get_gpu_availability(&devices) {
        Ok(GPUAvailability::Vacant) => {
            let devices_str = devices
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(", ");

            println!(
                "{} GPU {} is available!",
                "OK:".bold().green(),
                &devices_str
            );
            if args.last.len() > 0 {
                // Execute the command
                std::process::Command::new(&args.last[0])
                    .args(&args.last[1..])
                    .env("CUDA_VISIBLE_DEVICES", devices_str)
                    .spawn()
                    .expect("Failed to execute command");
            }
            ExitCode::SUCCESS
        }
        Ok(GPUAvailability::Occupied(status)) => {
            println!("{} GPU is currently in use", "ERROR:".bold().red());
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
