use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Mul;

use num::Num;
use rand::distributions::Distribution;
use rand::distributions::Standard;
use rand::Rng;

/// A 2-dimensional array used for neural network operations
#[derive(Debug, Default, Clone)]
pub struct Matrix<T>
where
    T: MatrixItem,
{
    /// The number of columns.
    pub cols: usize,
    /// The number of rows.
    pub rows: usize,
    /// The data in a single flat dynamic array.
    pub items: Vec<T>,
}

impl<T> Matrix<T>
where
    T: MatrixItem,
{
    /// Create a new matrix given the columns and rows.
    pub fn new<I: Into<usize>>(cols: I, rows: I) -> Self {
        let (cols, rows) = (cols.into(), rows.into());
        let items = vec![T::zero(); cols * rows];

        Self { cols, rows, items }
    }

    /// Create a new matrix from an existing `Vec<T>`.
    pub fn with_items<I: Into<usize>, J: Into<Vec<T>>>(items: J, cols: I, rows: I) -> Self {
        let (cols, rows) = (cols.into(), rows.into());
        let items = items.into();

        Self { cols, rows, items }
    }

    /// Resize the matrix and initializing everything to T::zero()
    pub fn resize<I: Into<usize>>(&mut self, cols: I, rows: I) {
        self.cols = cols.into();
        self.rows = rows.into();

        self.items.resize(self.cols * self.rows, T::default());
    }

    /// Overwrite the stuff inside the matrix.
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
    /// Randomize the stuff inside the matrix.
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

impl MatrixItem for f32 {}
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

    fn index(&self, (col, row): (usize, usize)) -> &Self::Output {
        if col >= self.cols || row >= self.rows {
            panic!("Index out of bounds while indexing matrix.\nMatrix[{col}, {row}]\n{self:?}");
        }

        &self.items[row * self.cols + col]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T>
where
    T: MatrixItem,
{
    fn index_mut(&mut self, (col, row): (usize, usize)) -> &mut Self::Output {
        if col >= self.cols || row >= self.rows {
            panic!("Index out of bounds while indexing matrix.\nMatrix[{col}, {row}]\n{self:?}");
        }

        &mut self.items[row * self.cols + col]
    }
}
