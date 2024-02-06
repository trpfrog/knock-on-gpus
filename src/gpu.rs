use std::error::Error;
use nvml_wrapper::Nvml;

pub(crate) fn parse_cuda_visible_devices() -> Option<Vec<u32>> {
    match std::env::var("CUDA_VISIBLE_DEVICES") {
        Ok(visible_devices) => {
            let visible_devices: Vec<u32> = visible_devices
                .split(',')
                .map(|s| s.parse::<u32>().unwrap())
                .collect();
            Some(visible_devices)
        }
        Err(_) => None,
    }
}

pub(crate) fn is_gpu_using(visible_devices: Option<Vec<u32>>) -> Result<bool, Box<dyn Error>> {
    let nvml = Nvml::init()?;
    let device_count = nvml.device_count()?;

    for i in 0..device_count {
        // If visible_devices is passed, only check those devices
        if let Some(visible_devices) = &visible_devices {
            if !visible_devices.contains(&i) {
                continue;
            }
        }

        let device = nvml.device_by_index(i)?;
        let used_memory_in_bytes = device.memory_info()?.used; // Currently 1.63/6.37 GB used on my system
        let utilization = device.utilization_rates()?; // 0% on my system

        // 100 MB = 100 * 1024 * 1024 bytes
        const BORDER_100_MB: u64 = 100 * 1024 * 1024;

        let device_is_using =
                used_memory_in_bytes > BORDER_100_MB ||
                utilization.gpu > 50 ||
                utilization.memory > 50;

        if device_is_using {
            return Ok(true);
        }
    }

    return Ok(false);
}
