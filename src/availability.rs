use anyhow::{bail, Context, Result};
use itertools::Itertools;
use log::debug;
use nvml_wrapper::Nvml;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GPUStatus {
    pub id: u32,
    pub used_memory: u64,
    pub gpu_utilization: u32,
    pub memory_utilization: u32,
    pub is_vacant: bool,
}

#[derive(Debug, Clone)]
pub enum GPUAvailability<T> {
    Vacant(T),
    Occupied(T),
}

fn validate_num_requested(num_requested: usize, devices: &Vec<u32>) -> Result<()> {
    if num_requested > devices.len() {
        bail!(
            concat!(
                "The number of requested GPUs is greater than the number of visible devices.\n",
                "Visible devices: {}\n",
                "The number of GPUs you requested: {}"
            ),
            devices.len(),
            num_requested
        );
    }
    Ok(())
}

pub fn get_gpu_availability(
    devices: &Vec<u32>,
    memory_border_mib: f32,
    num_requested: Option<usize>,
) -> Result<GPUAvailability<Vec<GPUStatus>>> {
    let nvml = Nvml::init()?;
    let memory_border_mib = memory_border_mib;
    let memory_border_bytes = (memory_border_mib * 1024.0 * 1024.0) as u64;

    if let Some(n) = num_requested {
        validate_num_requested(n, devices)?;
    }

    let status = devices
        .iter()
        .map(|&i| {
            nvml.device_by_index(i)
                .with_context(|| format!("Failed to get device by index {}", i))
        })
        .map_ok(|device| {
            let utilization = device.utilization_rates()?;
            let used_memory = device.memory_info()?.used;
            Ok(GPUStatus {
                id: device.index()?,
                used_memory,
                gpu_utilization: utilization.gpu,
                memory_utilization: utilization.memory,
                is_vacant: used_memory < memory_border_bytes,
            })
        })
        .flatten()
        .collect::<Result<Vec<GPUStatus>>>()?;

    debug!("GPU status:\n{:#?}", status);

    if let Some(n) = num_requested {
        let fullfilled = status.iter().filter(|s| s.is_vacant).count() >= n;
        if fullfilled {
            let status = status.into_iter().filter(|s| s.is_vacant).take(n).collect();
            debug!("Selected:\n{:#?}", status);
            Ok(GPUAvailability::Vacant(status))
        } else {
            Ok(GPUAvailability::Occupied(status))
        }
    } else {
        let is_all_vacant: bool = status.iter().all(|s| s.is_vacant);
        if is_all_vacant {
            Ok(GPUAvailability::Vacant(status))
        } else {
            Ok(GPUAvailability::Occupied(status))
        }
    }
}
