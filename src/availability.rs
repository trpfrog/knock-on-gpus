use nvml_wrapper::Nvml;
use std::error::Error;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GPUStatus {
    pub id: u32,
    pub used_memory: u64,
    pub gpu_utilization: u32,
    pub memory_utilization: u32,
}

#[derive(Debug, Clone)]
pub enum GPUAvailability {
    Vacant,
    Occupied(GPUStatus),
}

pub fn get_gpu_availability(
    devices: &Vec<u32>,
    memory_border: Option<f32>,
) -> Result<GPUAvailability, Box<dyn Error>> {
    let nvml = Nvml::init()?;

    for &i in devices {
        let device = nvml.device_by_index(i)?;
        let used_memory_in_bytes = device.memory_info()?.used;
        let utilization = device.utilization_rates()?;

        // 300 MB = 300 * 1024 * 1024 bytes

        let memory_border = memory_border.unwrap_or(300.0);
        let device_is_using = used_memory_in_bytes > (memory_border * 1024.0 * 1024.0) as u64
            || utilization.gpu > 50
            || utilization.memory > 50;

        if device_is_using {
            return Ok(GPUAvailability::Occupied(GPUStatus {
                id: i,
                used_memory: used_memory_in_bytes,
                gpu_utilization: utilization.gpu,
                memory_utilization: utilization.memory,
            }));
        }
    }

    Ok(GPUAvailability::Vacant)
}
