use anyhow::{Context, Result};
use itertools::Itertools;
use nvml_wrapper::Nvml;

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
    Vacant(Vec<GPUStatus>),
    Occupied(Vec<GPUStatus>),
}

pub fn get_gpu_availability(devices: &Vec<u32>, memory_border_mib: f32) -> Result<GPUAvailability> {
    let nvml = Nvml::init()?;
    let memory_border_mib = memory_border_mib;
    let memory_border_bytes = (memory_border_mib * 1024.0 * 1024.0) as u64;

    let status = devices
        .iter()
        .map(|&i| {
            nvml.device_by_index(i)
                .with_context(|| format!("Failed to get device by index {}", i))
        })
        .map_ok(|device| {
            let utilization = device.utilization_rates()?;
            Ok(GPUStatus {
                id: device.index()?,
                used_memory: device.memory_info()?.used,
                gpu_utilization: utilization.gpu,
                memory_utilization: utilization.memory,
            })
        })
        .flatten()
        .collect::<Result<Vec<GPUStatus>>>()?;

    let is_vacant = status.iter().any(|s| {
        s.used_memory < memory_border_bytes && s.gpu_utilization < 20 && s.memory_utilization < 20
    });

    if is_vacant {
        Ok(GPUAvailability::Vacant(status))
    } else {
        Ok(GPUAvailability::Occupied(status))
    }
}
