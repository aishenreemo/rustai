use std::cell::RefCell;

use num::Float;
use rand::distributions::Distribution;
use rand::distributions::Standard;
use rand::Rng;

use crate::matrix::Matrix;
use crate::matrix::MatrixItem;

/// The neural network.
#[derive(Debug)]
pub struct Network<T, U>
where
    T: NetworkItem,
    U: Activation<T>,
{
    /// All data of the neural network that is stored in a 1D matrix array.
    data: RefCell<Vec<Matrix<T>>>,
    arch: Box<[u8]>,
    activation: U,
}

impl<T> Network<T, ActivationVariant>
where
    T: NetworkItem,
{
    /// Create a new neural network with the default activation function.
    pub fn new(arch: &[u8]) -> Self {
        Self::with_activation(arch, ActivationVariant::default())
    }
}

impl<T, U> Network<T, U>
where
    T: NetworkItem,
    U: Activation<T>,
{
    /// Create a new neural network with the activation function.
    pub fn with_activation(arch: &[u8], activation: U) -> Self {
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

        Self {
            activation,
            arch,
            data,
        }
    }

    /// Get the index position of the matrix based on the network part that is specified.
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

    /// Apply the gradient to the original neural network data.
    pub fn learn(&mut self, learn_rate: T) {
        use NetworkPart::*;
        let mut data = self.data.borrow_mut();

        for i in 0..(self.arch.len() - 1) {
            let w_index = self.index(Weights, i);
            let b_index = self.index(Biases, i);
            let wg_index = self.index(WeightsGradient, i);
            let bg_index = self.index(BiasesGradient, i);

            for i in 0..data[w_index].items.len() {
                let g_value = data[wg_index].items[i];
                data[w_index].items[i] += -learn_rate * g_value;
            }

            for i in 0..data[b_index].items.len() {
                let g_value = data[bg_index].items[i];
                data[b_index].items[i] += -learn_rate * g_value;
            }
        }
    }

    /// Clone the output matrix
    pub fn output(&self) -> Matrix<T> {
        use NetworkPart::Activations;
        self.data.borrow()[self.index(Activations, self.arch.len() - 1)].clone()
    }

    /// Forwards the input specified to get an output.
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

            for item in data[activation_index].items.iter_mut() {
                *item = self.activation.activate(*item);
            }
        }
    }

    /// Compute for the loss function.
    pub fn cost(&mut self, tdata: &Matrix<T>) -> T {
        let input_cols = self.arch[0] as usize;
        let output_cols = self.arch[self.arch.len() - 1] as usize;
        if input_cols + output_cols != tdata.cols {
            panic!("Input mismatch while calculating cost. Training data corrupted.");
        }

        use NetworkPart::*;

        let mut result = T::zero();

        for i in 0..tdata.rows {
            let start = i * tdata.cols;
            let end = start + input_cols;
            let irow = &tdata.items[start..end];
            self.forward(irow);

            let start = i * tdata.cols + input_cols;
            let end = start + output_cols;
            let expected = &tdata.items[start..end];

            let start = self.index(Activations, self.arch.len() - 1);
            let predicted = &self.data.borrow()[start].items;

            for i in 0..output_cols {
                let diff = predicted[i] - expected[i];
                result += diff * diff;
            }
        }

        result
    }

    /// Backpropagate and get the gradient.
    pub fn backpropagate(&mut self, tdata: &Matrix<T>) {
        use NetworkPart::*;

        for i in 0..(self.arch.len() - 1) {
            let mut data = self.data.borrow_mut();
            let weights_g = self.index(WeightsGradient, i);
            let biases_g = self.index(BiasesGradient, i);

            for item in data[weights_g].items.iter_mut() {
                *item = T::zero();
            }

            for item in data[biases_g].items.iter_mut() {
                *item = T::zero();
            }
        }

        let input_cols = self.arch[0] as usize;
        let output_cols = self.arch[self.arch.len() - 1] as usize;

        for i in 0..tdata.rows {
            let start = i * tdata.cols;
            let end = start + input_cols;
            let irow = &tdata.items[start..end];
            self.forward(irow);

            let mut data = self.data.borrow_mut();
            for i in 0..self.arch.len() {
                let activations_g_index = self.index(ActivationsGradient, i);

                for item in data[activations_g_index].items.iter_mut() {
                    *item = T::zero();
                }
            }

            let start = i * tdata.cols + input_cols;
            let end = start + output_cols;
            let expected = &tdata.items[start..end];

            let predicted_index = self.index(Activations, self.arch.len() - 1);
            let output_g_index = self.index(ActivationsGradient, self.arch.len() - 1);

            for (i, &e) in expected.iter().enumerate() {
                let p = data[predicted_index][(i, 0)];
                data[output_g_index][(i, 0)] = T::from(2.0).unwrap() * (p - e);
            }

            for j in (1..self.arch.len()).rev() {
                let a_index = self.index(Activations, j);
                let ag_index = self.index(ActivationsGradient, j);

                let pa_index = self.index(Activations, j - 1);
                let pw_index = self.index(Weights, j - 1);

                let bg_index = self.index(BiasesGradient, j - 1);
                let wg_index = self.index(WeightsGradient, j - 1);
                let pag_index = self.index(ActivationsGradient, j - 1);

                for k in 0..data[a_index].cols {
		    let g = data[ag_index][(k, 0)];
		    let n = data[a_index][(k, 0)];
                    let d = self.activation.differentiate(n);

                    data[bg_index][(k, 0)] += g * d;

                    for l in 0..data[pa_index].cols {
                        let pn = data[pa_index][(l, 0)];
                        let w = data[pw_index][(k, l)];

                        data[wg_index][(k, l)] += g * d * pn;
                        data[pag_index][(l, 0)] += g * d * w;
                    }
                }
            }
        }

        for i in 0..(self.arch.len() - 1) {
            let mut data = self.data.borrow_mut();
            let weights_g_index = self.index(WeightsGradient, i);
            let biases_g_index = self.index(BiasesGradient, i);

            let n = T::from(1.0 / tdata.rows as f64).unwrap();

            for item in data[weights_g_index].items.iter_mut() {
                *item = (*item) * n;
            }

            for item in data[biases_g_index].items.iter_mut() {
                *item = (*item) * n;
            }
        }
    }
}

impl<T, U> Network<T, U>
where
    T: NetworkItem,
    U: Activation<T>,
    Standard: Distribution<T>,
{
    /// Randomizes all matrices inside the network.
    pub fn randomize<R: Rng>(&mut self, rng: &mut R) {
        use NetworkPart::*;
        let mut data = self.data.borrow_mut();

        for i in 0..(self.arch.len() - 1) {
            data[self.index(Weights, i)].randomize(rng);
            data[self.index(Biases, i)].randomize(rng);
        }
    }
}

/// The whole network is divided into subparts.
#[allow(dead_code)]
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
impl NetworkItem for f32 {}
impl NetworkItem for f64 {}

/// Implement this trait to your struct to have custom activation functions.
pub trait Activation<T: NetworkItem> {
    fn activate(&self, x: T) -> T;
    fn differentiate(&self, x: T) -> T;
}

/// Builtin activation function options.
#[allow(dead_code)]
#[derive(Default, Debug)]
pub enum ActivationVariant {
    #[default]
    Identity,
    Sigmoid,
}

#[allow(dead_code)]
impl<T> Activation<T> for ActivationVariant
where
    T: NetworkItem,
{
    /// Activation function.
    fn activate(&self, x: T) -> T {
        use ActivationVariant::*;
        let x = x.to_f64().expect("Unable to convert NetworkItem to primitive float.");
        let y = match self {
            &Identity => x,
            &Sigmoid => 1.0 / (1.0 + (-x).exp()),
        };

        T::from(y).expect("Unable to convert primitive float to NetworkItem.")
    }

    /// Derivative function.
    fn differentiate(&self, x: T) -> T {
        use ActivationVariant::*;
        let x = x.to_f64().expect("Unable to convert NetworkItem to primitive float.");

        let y = match self {
            &Identity => 1.0,
            &Sigmoid => x * (1.0 - x),
        };

        T::from(y).expect("Unable to convert primitive float to NetworkItem.")
    }
}
