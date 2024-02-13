use nvml_wrapper::Nvml;
use std::error::Error;

/// Returns all device IDs even if CUDA_VISIBLE_DEVICES is set
pub(crate) fn get_all_devices() -> Result<Vec<u32>, Box<dyn Error>> {
    let nvml = Nvml::init()?;
    let device_count = nvml.device_count()?;
    return Ok((0..device_count).collect());
}

/// Parses the CUDA_VISIBLE_DEVICES string
pub(crate) fn parse_cuda_visible_devices(devices_str: &str) -> Result<Vec<u32>, String> {
    let devices: Result<Vec<u32>, String> = devices_str
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<u32>()
                .map_err(|_| format!("Invalid device number: {}", s))
        })
        .collect();

    let nvml = Nvml::init().unwrap();
    let device_count = nvml.device_count().unwrap();

    return match devices {
        Ok(devices) if devices.iter().any(|d| *d >= device_count) => {
            Err(format!("Invalid device number: {}", devices_str))
        }
        Ok(devices) => Ok(devices),
        Err(e) => Err(e.to_string()),
    };
}

/// Returns all available devices using the CUDA_VISIBLE_DEVICES environment variable
pub(crate) fn get_visible_devices() -> Result<Vec<u32>, Box<dyn Error>> {
    let device_str = std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or("".to_string());
    return if device_str.is_empty() {
        get_all_devices()
    } else {
        Ok(parse_cuda_visible_devices(&device_str)?)
    };
}

pub(crate) fn pick_devices(pick_idx: &Vec<u32>, from_devices: &Vec<u32>) -> Vec<u32> {
    pick_idx
        .iter()
        .enumerate()
        .filter(|(i, _)| from_devices.contains(&(*i as u32)))
        .map(|(_, v)| *v)
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
            parse_cuda_visible_devices("a").unwrap_err(),
            "Invalid device number: a"
        );
        assert_eq!(
            parse_cuda_visible_devices("a,").unwrap_err(),
            "Invalid device number: a"
        );
        assert_eq!(
            parse_cuda_visible_devices("0,a").unwrap_err(),
            "Invalid device number: a"
        );
        assert_eq!(
            parse_cuda_visible_devices("0,a,").unwrap_err(),
            "Invalid device number: a"
        );
        assert_eq!(
            parse_cuda_visible_devices("0,x,2,3").unwrap_err(),
            "Invalid device number: x"
        );
        assert_eq!(
            parse_cuda_visible_devices("0,1,2,3,x").unwrap_err(),
            "Invalid device number: x"
        );
    }

    #[test]
    fn test_pick_devices() {
        assert_eq!(
            pick_devices(&vec![0, 1, 2], &vec![0, 1, 2, 3]),
            vec![0, 1, 2]
        );
        assert_eq!(pick_devices(&vec![0, 1, 2], &vec![0, 1]), vec![0, 1]);
        assert_eq!(pick_devices(&vec![0, 1, 2], &vec![0, 2]), vec![0, 2]);
        assert_eq!(pick_devices(&vec![0, 1, 2], &vec![1, 2]), vec![1, 2]);
        assert_eq!(pick_devices(&vec![0, 1, 2], &vec![1]), vec![1]);
        assert_eq!(pick_devices(&vec![0, 1, 2], &vec![2]), vec![2]);
        assert_eq!(pick_devices(&vec![0, 1, 2], &vec![3]), vec![]);
    }
}
