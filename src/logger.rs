use colored::Colorize;
use env_logger;
use std::env;
use std::io::Write;

pub(crate) fn init_logger() {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "info");
    }
    env_logger::Builder::from_default_env()
        .format(|buf, record| {
            let level = match record.level() {
                log::Level::Error => "ERROR:".red().bold(),
                log::Level::Warn => "WARNING:".yellow().bold(),
                log::Level::Info => "INFO:".green().bold(),
                log::Level::Debug => "DEBUG:".blue().bold(),
                log::Level::Trace => "TRACE:".purple().bold(),
            };
            writeln!(buf, "{} {}\n", level, record.args())
        })
        .init();
}
