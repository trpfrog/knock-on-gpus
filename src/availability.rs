use nvml_wrapper::Nvml;
use std::error::Error;

#[derive(Debug, Clone)]
pub struct GPUStatus {
    id: u32,
    gpu_utilization: u32,
    memory_utilization: u32,
}

#[derive(Debug, Clone)]
pub enum GPUAvailability {
    Vacant,
    Occupied(GPUStatus),
}

pub fn get_gpu_availability(devices: &Vec<u32>) -> Result<GPUAvailability, Box<dyn Error>> {
    let nvml = Nvml::init()?;

    for &i in devices {
        let device = nvml.device_by_index(i)?;
        let used_memory_in_bytes = device.memory_info()?.used;
        let utilization = device.utilization_rates()?;

        // 100 MB = 100 * 1024 * 1024 bytes
        const BORDER_100_MB: u64 = 100 * 1024 * 1024;

        let device_is_using =
            used_memory_in_bytes > BORDER_100_MB || utilization.gpu > 50 || utilization.memory > 50;

        if device_is_using {
            return Ok(GPUAvailability::Occupied(GPUStatus {
                id: i,
                gpu_utilization: utilization.gpu,
                memory_utilization: utilization.memory,
            }));
        }
    }

    return Ok(GPUAvailability::Vacant);
}
