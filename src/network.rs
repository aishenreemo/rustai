use anyhow::bail;
use anyhow::Context;
use anyhow::Ok;
use anyhow::Result;

use crate::activation::Activation;
use crate::matrix::Matrix;

#[derive(Debug)]
pub struct Network<T> {
    activation: Activation<T>,
    layers: Box<[usize]>,
    data: NetworkData<T>,
}

impl<T> Network<T>
where
    T: Default + Clone + Copy,
{
    pub fn new(arch: &[usize]) -> Result<Self> {
        Network::<T>::is_valid_arch(arch).context("Cannot construct neural network.")?;

        Ok(Self {
            activation: Activation::default(),
            layers: arch.into(),
            data: NetworkData::new(arch),
        })
    }

    pub fn with_activation(arch: &[usize], activation: Activation<T>) -> Result<Self> {
        Network::<T>::is_valid_arch(arch)
            .context("Cannot construct neural network with activation.")?;

        Ok(Self {
            activation,
            layers: arch.into(),
            data: NetworkData::new(arch),
        })
    }
}

impl<T> Network<T> {
    fn is_valid_arch(arch: &[usize]) -> Result<()> {
        use std::cmp::Ordering::*;

        match arch.len().cmp(&2) {
            Equal | Greater => Ok(()),
            Less => bail!(
                "Invalid architecture, expected to have atleast 2 layers, got {}.",
                arch.len()
            ),
        }
    }
}

#[derive(Debug)]
struct NetworkData<T> {
    input: Matrix<T>,
    layers: Box<[NetworkLayer<T>]>,
}

impl<T> NetworkData<T>
where
    T: Default + Clone + Copy,
{
    fn new(arch: &[usize]) -> Self {
        Self {
            input: Matrix::new(arch[0], 1),
            layers: NetworkLayer::new(arch),
        }
    }
}

#[derive(Debug)]
struct NetworkLayer<T> {
    w: Matrix<T>,
    b: Matrix<T>,
    a: Matrix<T>,
}

impl<T> NetworkLayer<T> {
    fn new(arch: &[usize]) -> Box<[Self]> {
        let layers = Vec::with_capacity(arch.len());

        layers.into()
    }
}
