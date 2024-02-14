use std::collections::HashSet;

use anyhow::{anyhow, Context, Result};
use itertools::Itertools;
use nvml_wrapper::Nvml;

/// Checks if CUDA is available
pub(crate) fn is_cuda_available() -> bool {
    let nvml = Nvml::init();
    log::debug!("CUDA is available: {}", nvml.is_ok());
    nvml.is_ok()
}

/// Returns all device IDs even if CUDA_VISIBLE_DEVICES is set
pub(crate) fn get_all_devices() -> Result<Vec<u32>> {
    let nvml = Nvml::init()?;
    let device_count = nvml.device_count()?;
    Ok((0..device_count).collect())
}

/// Parses the CUDA_VISIBLE_DEVICES string
pub(crate) fn parse_cuda_visible_devices(devices_str: &str) -> Result<Vec<u32>> {
    let nvml =
        Nvml::init().context("Failed to initialize NVML. Probably no NVIDIA GPU is installed.")?;
    let device_count = nvml.device_count()?;
    parse_cuda_visible_devices_with_device_count(devices_str, device_count)
}

fn parse_cuda_visible_devices_with_device_count(
    devices_str: &str,
    device_count: u32,
) -> Result<Vec<u32>> {
    devices_str
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<u32>()
                .with_context(|| format!("Invalid device number: {}", s))
        })
        .map(|result| match result {
            Ok(device) if device >= device_count => {
                Err(anyhow!("Device number {} is out of range", device))
            }
            other => other,
        })
        .collect::<Result<HashSet<u32>>>()? // Unique
        .into_iter()
        .sorted()
        .map(Result::Ok) // Convert to Result
        .collect::<Result<Vec<u32>>>()
}

/// Returns all available devices using the CUDA_VISIBLE_DEVICES environment variable
pub(crate) fn get_visible_devices(cuda_visible_devices_env_key: Option<&str>) -> Result<Vec<u32>> {
    let key = cuda_visible_devices_env_key.unwrap_or("CUDA_VISIBLE_DEVICES");
    if let Ok(devices_str) = std::env::var(key) {
        parse_cuda_visible_devices(&devices_str)
    } else {
        get_all_devices()
    }
}

/// Picks devices from the list of available devices
pub(crate) fn pick_devices(pick_idx: &Vec<u32>, from_devices: &Vec<u32>) -> Result<Vec<u32>> {
    pick_idx
        .iter()
        .map(|i| {
            from_devices
                .get(*i as usize)
                .copied()
                .with_context(|| format!("Index {} is out of range.", i))
        })
        .collect::<Result<Vec<u32>>>()?
        .into_iter()
        .sorted()
        .map(Result::Ok)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cuda_visible_devices() {
        let f = |s, n| parse_cuda_visible_devices_with_device_count(s, n).unwrap();
        assert_eq!(f("0,1,2", 4), vec![0, 1, 2]);
        assert_eq!(f("0,1,2,", 4), vec![0, 1, 2]);
        assert_eq!(f("0,1,2,3", 4), vec![0, 1, 2, 3]);
        assert_eq!(f(",2,3,", 4), vec![2, 3]);
        assert_eq!(f("0", 4), vec![0]);
        assert_eq!(f("0,", 4), vec![0]);
        assert_eq!(f("", 4), vec![]);
        assert_eq!(f("0,1,2,3", 8), vec![0, 1, 2, 3]);
        assert_eq!(f("0,1,2,3,4,5,6,7", 8), vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_parse_cuda_visible_devices_error() {
        let f = &parse_cuda_visible_devices_with_device_count;
        assert!(f("a,", 4).is_err());
        assert!(f("0,a", 4).is_err());
        assert!(f("0,a,", 4).is_err());
        assert!(f("0,x,2,3", 4).is_err());
        assert!(f("0,1,2,3,x", 4).is_err());
        assert!(f("0,1,-1", 4).is_err());
        assert!(f("0,1,2,4", 4).is_err());
        assert!(f("0,1,2,8,", 8).is_err());
    }

    #[test]
    fn test_pick_device() {
        let f = |pick_idx: Vec<u32>, from_devices: Vec<u32>| {
            pick_devices(&pick_idx, &from_devices).unwrap()
        };

        assert_eq!(f(vec![], vec![]), vec![]);
        assert_eq!(f(vec![], vec![0, 1, 2]), vec![]);
        assert_eq!(f(vec![0], vec![0, 1, 2]), vec![0]);
        assert_eq!(f(vec![0, 1], vec![2, 1]), vec![1, 2]);
        assert_eq!(f(vec![0, 1, 2], vec![2, 1, 0]), vec![0, 1, 2]);
        assert_eq!(f(vec![1], vec![5, 6, 7]), vec![6]);

        let f = |pick_idx: Vec<u32>, from_devices: Vec<u32>| {
            pick_devices(&pick_idx, &from_devices)
                .unwrap_err()
                .to_string()
        };

        assert_eq!(f(vec![0, 1, 2], vec![]), "Index 0 is out of range.");
        assert_eq!(f(vec![0, 1, 2], vec![0]), "Index 1 is out of range.");
        assert_eq!(f(vec![0, 1, 2], vec![0, 1]), "Index 2 is out of range.");
    }
}
