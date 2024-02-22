use colored::Colorize;
use env_logger;
use std::env;
use std::io::Write;

pub(crate) fn init_logger(default_level: log::Level) {
    if env::var("RUST_LOG").is_err() {
        env::set_var(
            "RUST_LOG",
            match default_level {
                log::Level::Error => "error",
                log::Level::Warn => "warn",
                log::Level::Info => "info",
                log::Level::Debug => "debug",
                log::Level::Trace => "trace",
            },
        );
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
