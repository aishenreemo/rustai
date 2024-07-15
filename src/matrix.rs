use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Mul;

use num::Num;
use rand::distributions::Distribution;
use rand::distributions::Standard;
use rand::Rng;

use crate::network::Activation;

#[derive(Debug, Default, Clone)]
pub struct Matrix<T>
where
    T: MatrixItem,
{
    pub cols: usize,
    pub rows: usize,
    pub items: Vec<T>,
}

impl<T> Matrix<T>
where
    T: MatrixItem,
{
    pub fn new<I: Into<usize>>(cols: I, rows: I) -> Self {
        let (cols, rows) = (cols.into(), rows.into());
        let items = vec![T::default(); cols * rows];

        Self { cols, rows, items }
    }

    pub fn with_items<I: Into<usize>, J: Into<Vec<T>>>(items: J, cols: I, rows: I) -> Self {
        let (cols, rows) = (cols.into(), rows.into());
        let items = items.into();

        Self { cols, rows, items }
    }

    pub fn resize<I: Into<usize>>(&mut self, cols: I, rows: I) {
        self.cols = cols.into();
        self.rows = rows.into();

        self.items.resize(self.cols * self.rows, T::default());
    }

    pub fn set_data(&mut self, input: &[T]) {
        if input.len() != self.cols {
            panic!("Input mismatch while setting data to matrix.");
        }

        for (item, input_item) in self.items.iter_mut().zip(input.iter()) {
            *item = *input_item;
        }
    }
}

impl<T> Matrix<T>
where
    T: MatrixItem,
    Standard: Distribution<T>,
{
    pub fn randomize<R: Rng>(&mut self, rng: &mut R) {
        for item in self.items.iter_mut() {
            *item = rng.gen();
        }
    }
}

pub trait MatrixItem
where
    Self: std::fmt::Debug + Default + Clone + Copy + Num + AddAssign,
{
}

impl MatrixItem for f64 {}

impl<T> Mul for Matrix<T>
where
    T: MatrixItem,
{
    type Output = Matrix<T>;

    fn mul(self, other: Self) -> Self::Output {
        if self.cols != other.rows {
            panic!("Invalid matrix dimensions, cannot multiply matrix.");
        }

        let mut result = Matrix::new(other.cols, self.rows);

        for i in 0..other.cols {
            for j in 0..self.rows {
                for k in 0..self.cols {
                    result[(i, j)] += self[(k, j)] * other[(i, k)];
                }
            }
        }

        result
    }
}

impl<T> Add for Matrix<T>
where
    T: MatrixItem,
{
    type Output = Matrix<T>;

    fn add(self, other: Self) -> Self::Output {
        if self.cols != other.cols || self.rows != other.rows {
            panic!("Invalid matrix dimensions, cannot add matrix.");
        }

        let mut result = Matrix::new(self.cols, self.rows);

        for i in 0..other.cols {
            for j in 0..self.rows {
                result[(i, j)] = self[(i, j)] + other[(i, j)];
            }
        }

        result
    }
}

impl<T> Index<(usize, usize)> for Matrix<T>
where
    T: MatrixItem,
{
    type Output = T;

    fn index(&self, (cols, rows): (usize, usize)) -> &Self::Output {
        if cols >= self.cols || rows >= self.rows {
            panic!("Index out of bounds while indexing matrix.");
        }

        &self.items[rows * self.cols + cols]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T>
where
    T: MatrixItem,
{
    fn index_mut(&mut self, (cols, rows): (usize, usize)) -> &mut Self::Output {
        if cols >= self.cols || rows >= self.rows {
            panic!("Index out of bounds while indexing matrix.");
        }

        &mut self.items[rows * self.cols + cols]
    }
}
