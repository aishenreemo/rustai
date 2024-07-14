#[derive(Debug)]
pub struct Matrix<T> {
    cols: usize,
    rows: usize,
    items: Vec<T>,
}

impl<T> Matrix<T>
where
    T: Default + Clone + Copy,
{
    pub fn new(cols: usize, rows: usize) -> Self {
        Self {
            cols,
            rows,
            items: vec![T::default(); cols * rows],
        }
    }
}
