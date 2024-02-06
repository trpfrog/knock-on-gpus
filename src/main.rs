mod gpu;

use std::process::{ExitCode};
use gpu::is_gpu_using;
use clap::Parser;
use colored::{Colorize};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Verbose mode
    #[arg(long)]
    verbose: bool,

    /// Silence mode
    #[arg(long)]
    silent: bool,

    /// Check all devices
    #[arg(short, long)]
    all: bool,
}


fn main() -> ExitCode {
    let args = Args::parse();
    assert!(!(args.silent && args.verbose), "Cannot be both silent and verbose");

    let visible_devices = if args.all {
        None
    } else {
        gpu::parse_cuda_visible_devices()
    };

    return match is_gpu_using(visible_devices) {
        Ok(false) => {
            if args.verbose {
                println!("{} GPU is available", "OK:".bold().green());
            }
            ExitCode::SUCCESS
        },
        Ok(true) => {
            if !args.silent {
                println!("{} GPU is currentry in use", "ERROR:".bold().red());
                println!("See `nvidia-smi` for more information");
            }
            ExitCode::FAILURE
        }
        Err(e) => {
            // println!("Error has occurred while checking GPU usage: {}", e);
            eprintln!("{} Error has occurred while checking GPU usage:", "ERROR".bold().red());
            eprintln!("{}", e);
            ExitCode::FAILURE
        }
    };
}
