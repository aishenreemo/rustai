use std::cell::RefCell;

use num::Float;
use rand::distributions::Distribution;
use rand::distributions::Standard;
use rand::Rng;

use crate::matrix::Matrix;
use crate::matrix::MatrixItem;

#[derive(Debug)]
pub struct Network<T>
where
    T: NetworkItem,
{
    activation: Activation<T>,
    arch: Box<[u8]>,
    data: RefCell<Vec<Matrix<T>>>,
}

impl<T> Network<T>
where
    T: NetworkItem,
{
    pub fn new(arch: &[u8]) -> Self {
        if arch.len() < 2 {
            panic!(
                "Cannot construct a neural network, expects atleast 2 layers got {}",
                arch.len()
            );
        }

        let length = 6 * arch.len() - 4;
        let mut data = Vec::with_capacity(length);
        data.resize(length, Matrix::default());

        for i in 0..2 {
            data[i * arch.len()].resize(arch[0], 1);
            for (j, pair) in arch.windows(2).enumerate() {
                data[i * arch.len() + j + 1].resize(pair[1], 1);
                data[2 * arch.len() + i * (arch.len() - 1) + j].resize(pair[1], pair[0]);
                data[2 * arch.len() + (2 + i) * (arch.len() - 1) + j].resize(pair[1], 1);
            }
        }

        let arch: Box<[u8]> = arch.into();
        let data: RefCell<Vec<Matrix<T>>> = RefCell::new(data);
        let activation: Activation<T> = Activation::default();

        Self {
            activation,
            arch,
            data,
        }
    }

    pub fn with_activation(arch: &[u8], activation: Activation<T>) -> Self {
        let mut network = Network::new(arch);
        network.activation = activation;

        network
    }

    fn index(&self, part: NetworkPart, layer_index: usize) -> usize {
        use NetworkPart::*;

        let arch_len = self.arch.len();
        let wb_len = arch_len - 1;

        if ((part == Activations || part == ActivationsGradient) && layer_index >= arch_len) ||
            ((part != Activations && part != ActivationsGradient) && layer_index >= wb_len)
        {
            panic!("Index out of bounds while indexing neural network data.");
        }

        #[rustfmt::skip]
        match part {
            Activations         => 0 * arch_len + 0 * wb_len + layer_index,
            ActivationsGradient => 1 * arch_len + 0 * wb_len + layer_index,
            Weights             => 2 * arch_len + 0 * wb_len + layer_index,
            WeightsGradient     => 2 * arch_len + 1 * wb_len + layer_index,
            Biases              => 2 * arch_len + 2 * wb_len + layer_index,
            BiasesGradient      => 2 * arch_len + 3 * wb_len + layer_index,
        }
    }
}

impl<T> Network<T>
where
    T: NetworkItem,
    f64: From<T>,
{
    pub fn forward(&mut self, input: &[T]) {
        use NetworkPart::*;

        let mut data = self.data.borrow_mut();
        let input_index = self.index(Activations, 0);

        data[input_index].set_data(input);

        for i in 0..(self.arch.len() - 1) {
            let prev_activation_index = self.index(Activations, i);
            let activation_index = self.index(Activations, i + 1);
            let weight_index = self.index(Weights, i);
            let biases_index = self.index(Biases, i);

            data[activation_index] = data[prev_activation_index].clone() *
                data[weight_index].clone() +
                data[biases_index].clone();

            let cols = data[activation_index].cols;
            let rows = data[activation_index].rows;
            for i in 0..(cols * rows) {
                let col = i % cols;
                let row = i / cols;

                let matrix = &mut data[activation_index];
                matrix[(col, row)] = self.activation.activate(matrix[(col, row)]);
            }
        }
    }
}

impl<T> Network<T>
where
    T: NetworkItem,
    Standard: Distribution<T>,
{
    pub fn randomize<R: Rng>(&mut self, rng: &mut R) {
        use NetworkPart::*;
        let mut data = self.data.borrow_mut();

        for i in 0..(self.arch.len() - 1) {
            data[self.index(Weights, i)].randomize(rng);
            data[self.index(Biases, i)].randomize(rng);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NetworkPart {
    ActivationsGradient,
    WeightsGradient,
    BiasesGradient,
    Activations,
    Weights,
    Biases,
}

pub trait NetworkItem: MatrixItem + Float {}
impl NetworkItem for f64 {}

#[derive(Default, Debug)]
pub enum Activation<T>
where
    T: NetworkItem,
{
    #[default]
    Identity,
    Sigmoid,
    Custom(fn(T) -> T, fn(T) -> T),
}

impl<T> Activation<T>
where
    T: NetworkItem,
    f64: From<T>,
{
    fn activate(&self, x: T) -> T {
        use Activation::*;
        match self {
            &Identity => x,
            &Sigmoid => T::from(1.0 / (1.0 + f64::from(x))).unwrap(),
            &Custom(f, ..) => f(x),
        }
    }

    fn derivative(&self, x: T) -> T {
        use Activation::*;
        match self {
            &Identity => x,
            &Sigmoid => T::from(1.0 / (1.0 + f64::from(x))).unwrap(),
            &Custom(.., d) => d(x),
        }
    }
}
