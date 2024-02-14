use anyhow::{anyhow, Context, Result};
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

    devices_str
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            let parse_result = s
                .parse::<u32>()
                .with_context(|| format!("Invalid device number: {}", s));
            match parse_result {
                Ok(device) if device < device_count => Ok(device),
                Ok(device) => Err(anyhow!("Device number {} is out of range", device)),
                Err(e) => Err(e),
            }
        })
        .collect()
}

/// Returns all available devices using the CUDA_VISIBLE_DEVICES environment variable
pub(crate) fn get_visible_devices() -> Result<Vec<u32>> {
    if let Ok(devices_str) = std::env::var("CUDA_VISIBLE_DEVICES") {
        parse_cuda_visible_devices(&devices_str)
    } else {
        get_all_devices()
    }
}

/// Picks devices from the list of available devices
pub(crate) fn pick_devices(pick_idx: &Vec<u32>, from_devices: &Vec<u32>) -> Result<Vec<u32>> {
    pick_idx
        .iter()
        .copied()
        .map(|i| {
            from_devices
                .get(i as usize)
                .copied()
                .with_context(|| format!("Index {} is out of range.", i))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cuda_visible_devices() {
        assert_eq!(parse_cuda_visible_devices("0,1,2").unwrap(), vec![0, 1, 2]);
        assert_eq!(parse_cuda_visible_devices("0,1,2,").unwrap(), vec![0, 1, 2]);
        assert_eq!(
            parse_cuda_visible_devices("0,1,2,3").unwrap(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(parse_cuda_visible_devices(",2,3,").unwrap(), vec![2, 3]);
        assert_eq!(parse_cuda_visible_devices("0").unwrap(), vec![0]);
        assert_eq!(parse_cuda_visible_devices("0,").unwrap(), vec![0]);
    }

    #[test]
    fn test_parse_cuda_visible_devices_error() {
        assert_eq!(
            parse_cuda_visible_devices("a").unwrap_err().to_string(),
            "Invalid device number: a"
        );
        assert_eq!(
            parse_cuda_visible_devices("a,").unwrap_err().to_string(),
            "Invalid device number: a"
        );
        assert_eq!(
            parse_cuda_visible_devices("0,a").unwrap_err().to_string(),
            "Invalid device number: a"
        );
        assert_eq!(
            parse_cuda_visible_devices("0,a,").unwrap_err().to_string(),
            "Invalid device number: a"
        );
        assert_eq!(
            parse_cuda_visible_devices("0,x,2,3")
                .unwrap_err()
                .to_string(),
            "Invalid device number: x"
        );
        assert_eq!(
            parse_cuda_visible_devices("0,1,2,3,x")
                .unwrap_err()
                .to_string(),
            "Invalid device number: x"
        );
    }

    #[test]
    fn test_pick_devices() {
        assert_eq!(
            pick_devices(&vec![0, 1, 2], &vec![0, 1, 2, 3]).unwrap(),
            vec![0, 1, 2]
        );
        assert_eq!(
            pick_devices(&vec![0, 1, 2, 3], &vec![0, 1, 2, 3]).unwrap(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            pick_devices(&vec![0, 2], &vec![0, 1, 2, 3]).unwrap(),
            vec![0, 2]
        );
        assert_eq!(
            pick_devices(&vec![0, 2], &vec![1, 3, 5]).unwrap(),
            vec![1, 5]
        );
    }
}
