use std::fs::read;
use std::io::{Error, ErrorKind, Result};
use std::path::Path;

pub struct Idx {
    pub dimensions: Vec<usize>,
    pub array: Vec<u8>,
}

unsafe fn load_i32(v: &[u8], i: usize) -> i32 {
    let res = 0i32;
    let size = std::mem::size_of::<i32>();
    for j in 0..size {
        *((&res as *const i32) as *mut u8).add(size - j - 1) = v[i + j];
    }
    res
} 

pub fn load_idx<T>(file_name: &str) -> Result<Idx> {
    let path = Path::new(file_name);
    println!("loading {}...", path.display());

    let bytes = read(path).unwrap();

    if !bytes[0] == 0 && bytes[1] == 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("Invalid file format: {}", file_name),
        ));
    }

    let dtype: u8 = bytes[2];

    let dtype_str = match dtype {
        0x08 => "u8",
        0x09 => "i8",
        0x0B => "i16",
        0x0C => "i32",
        0x0D => "f32",
        0x0E => "f64",
        _ => "INVALID TYPE",
    };

    let expected_dtype = core::any::type_name::<T>();

    if expected_dtype != dtype_str {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!(
                "File does not contain values of type {}: {}",
                expected_dtype, dtype_str
            ),
        ));
    };

    let n_dimensions = bytes[3] as usize;
    let mut dimensions = Vec::with_capacity(n_dimensions);

    for d in 0..n_dimensions {
        dimensions.push(unsafe { load_i32(&bytes, 4 + d * std::mem::size_of::<i32>()) as usize });
    }

    println!("dimensions: {:?}", dimensions);

    let values_start = 4 + n_dimensions * std::mem::size_of::<i32>();

    let array = bytes[values_start..].to_vec();

    Ok(Idx { dimensions, array })
}