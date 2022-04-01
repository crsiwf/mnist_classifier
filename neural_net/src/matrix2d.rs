use rand::Rng;
use rand_distr::StandardNormal;
use std::cmp::min;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

pub type Shape = (usize, usize);

#[derive(PartialEq)]
pub struct Matrix2D {
    shape: Shape,
    values: Vec<f32>,
}

impl Matrix2D {
    // constructors

    pub fn from_vec(shape: Shape, values: Vec<f32>) -> Self {
        Self { shape, values }
    }

    pub fn zeros(shape: Shape) -> Self {
        Self {
            shape,
            values: vec![0.0; shape.0 * shape.1],
        }
    }

    pub fn zeros_like(like: &Self) -> Self {
        let shape = like.shape;
        Self {
            shape,
            values: vec![0.0; shape.0 * shape.1],
        }
    }

    pub fn ones(shape: Shape) -> Self {
        Self {
            shape,
            values: vec![1.0; shape.0 * shape.1],
        }
    }

    pub fn ones_like(like: &Self) -> Self {
        let shape = like.shape;
        Self {
            shape,
            values: vec![1.0; shape.0 * shape.1],
        }
    }

    pub fn full(shape: Shape, value: f32) -> Self {
        Self {
            shape,
            values: vec![value; shape.0 * shape.1],
        }
    }

    pub fn random_std_normal(shape: Shape) -> Self {
        let size = shape.0 * shape.1;
        let mut values = Vec::with_capacity(size);

        for _ in 0..size {
            values.push(rand::thread_rng().sample(StandardNormal));
        }

        Self { shape, values }
    }

    pub fn random_uniform(shape: Shape, low: f32, high: f32) -> Self {
        let size = shape.0 * shape.1;
        let mut values = Vec::with_capacity(size);

        for _ in 0..size {
            values.push(rand::thread_rng().gen_range(low..high));
        }

        Self { shape, values }
    }

    pub fn diagonal(shape: Shape, value: f32) -> Self {
        let mut res = Self::zeros(shape);
        for i in 0..min(shape.0, shape.1) {
            res[(i, i)] = value;
        }
        res
    }

    pub fn diag_from_vec(shape: Shape, diagonal: Vec<f32>) -> Self {
        let mut res = Self::zeros(shape);
        for i in 0..min(shape.0, shape.1) {
            res[(i, i)] = diagonal[i];
        }
        res
    }

    pub unsafe fn from_raw_parts(shape: Shape, values: *mut f32) -> Self {
        let size = shape.0 * shape.1;
        Self {
            shape,
            values: Vec::from_raw_parts(values, size, size),
        }
    }

    // misc

    pub fn get_vec(&self) -> Vec<f32> {
        self.values.clone()
    }

    fn shape_to_vec_index(&self, shape: Shape) -> usize {
        shape.0 * self.columns() + shape.1
    }

    fn vec_index_to_shape(&self, index: usize) -> Shape {
        (index / self.columns(), index % self.columns())
    }

    pub fn size(&self) -> usize {
        self.rows() * self.columns()
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn rows(&self) -> usize {
        self.shape.0
    }

    pub fn columns(&self) -> usize {
        self.shape.1
    }

    // operations

    pub fn transpose(&self) -> Self {
        let mut res = Self::zeros((self.columns(), self.rows()));
        for row in 0..self.rows() {
            for col in 0..self.columns() {
                res[(col, row)] = self[(row, col)];
            }
        }
        res
    }

    pub fn reshape(&self, shape: Shape) -> Self {
        assert_eq!(
            self.size(),
            shape.0 * shape.1,
            "Cannot reshape to different size (from {:?} to {:?})",
            self.shape,
            shape
        );
        Self {
            shape,
            values: self.values.clone(),
        }
    }

    pub fn reshape_mut(&mut self, shape: Shape) {
        assert_eq!(
            self.size(),
            shape.0 * shape.1,
            "Cannot reshape to different size (from {:?} to {:?})",
            self.shape,
            shape
        );
        self.shape = shape;
    }

    pub fn map<F>(&self, f: F) -> Self
    where
        F: FnMut(f32) -> f32,
    {
        Self {
            shape: self.shape,
            values: self.get_vec().into_iter().map(f).collect(),
        }
    }

    pub fn map_mut<F>(&mut self, f: F)
    where
        F: FnMut(f32) -> f32,
    {
        self.values = self.get_vec().into_iter().map(f).collect();
    }

    pub fn filter<F>(&self, mut f: F) -> Self
    where
        F: FnMut(f32) -> bool,
    {
        self.map(|x_i| if f(x_i) {x_i} else {0.0})
    }

    pub fn filter_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(f32) -> bool,
    {
        self.map_mut(|x_i| if f(x_i) {x_i} else {0.0});
    }

    pub fn fold<F>(&self, init: f32, f: F) -> f32
    where
        F: FnMut(f32, f32) -> f32,
    {
        self.get_vec().into_iter().fold(init, f)
    }

    pub fn max(&self) -> f32 {
        self.fold(f32::NEG_INFINITY, f32::max)
    }

    pub fn argmax(&self) -> Shape {
        let mut x = f32::NEG_INFINITY;
        let mut xi = 0;
        for (i, &v) in self.values.iter().enumerate() {
            if v > x {
                x = v;
                xi = i;
            }
        }
        self.vec_index_to_shape(xi)
    }

    pub fn min(&self) -> f32 {
        self.fold(f32::NEG_INFINITY, f32::min)
    }

    pub fn argmin(&self) -> Shape {
        let mut x = f32::NEG_INFINITY;
        let mut xi = 0;
        for (i, &v) in self.values.iter().enumerate() {
            if v < x {
                x = v;
                xi = i;
            }
        }
        self.vec_index_to_shape(xi)
    }

    pub fn vstack(a: &Self, b: &Self) -> Self {
        assert_eq!(
            a.shape.1, b.shape.1,
            "Cannot vstack Matrices with different column counts."
        );

        let mut res = Self::zeros((a.rows() + b.rows(), a.columns()));
        for i in 0..a.size() {
            res.values[i] = a.values[i];
        }
        for i in 0..b.size() {
            res.values[a.size() + i] = b.values[i];
        }
        res
    }

    pub fn hstack(a: &Self, b: &Self) -> Self {
        assert_eq!(
            a.rows(),
            b.rows(),
            "Cannot vstack Matrices with different row counts."
        );

        let mut res = Self::zeros((a.rows(), a.columns() + b.columns()));
        for row in 0..res.rows() {
            for col_a in 0..a.columns() {
                res[(row, col_a)] = a[(row, col_a)];
            }
            for col_b in 0..b.columns() {
                let col_res = col_b + a.columns();
                res[(row, col_res)] = b[(row, col_b)];
            }
        }
        res
    }

    pub fn get_row(&self, row: usize) -> Self {
        assert!(
            row < self.rows(),
            "Row {} out of bounds for shape {:?}.",
            row,
            self.shape
        );

        let index_start = row * self.columns();
        let index_end = index_start + self.columns();

        Self {
            shape: (1, self.columns()),
            values: self.values[index_start..index_end].to_vec(),
        }
    }

    pub unsafe fn get_row_slice(&mut self, row: usize) -> &mut [f32] {
        let index_start = row * self.columns();
        let index_end = index_start + self.columns();

        &mut self.values[index_start..index_end]
    }

    pub fn get_column(&self, column: usize) -> Self {
        assert!(
            column < self.columns(),
            "Column {} out of bounds for shape {:?}.",
            column,
            self.shape
        );

        let mut column_vec = vec![0.0; self.rows()];
        for row in 0..self.rows() {
            column_vec[row] = self[(row, column)];
        }

        Self {
            shape: (self.rows(), 1),
            values: column_vec,
        }
    }


    pub fn slice_front(&self, n_columns: usize) -> Self {
        assert!(
            n_columns < self.columns(),
            "Cannot slice {} columns from Matrix with shape {:?}",
            n_columns,
            self.shape
        );

        let mut column_vec = vec![0.0; n_columns * self.rows()];
        for row in 0..self.rows() {
            for column in 0..n_columns {
                column_vec[row * n_columns + column] = self[(row, column)];
            }
        }

        Self {
            shape: (self.rows(), n_columns),
            values: column_vec,
        }
    }

    pub fn slice_back(&self, n_columns: usize) -> Self {
        assert!(
            n_columns < self.columns(),
            "Cannot slice {} columns from Matrix with shape {:?}",
            n_columns,
            self.shape
        );

        let column_start = self.columns() - n_columns;

        let mut column_vec = vec![0.0; n_columns * self.rows()];
        for row in 0..self.rows() {
            for column in 0..n_columns {
                column_vec[row * n_columns + column] = self[(row, column_start + column)];
            }
        }

        Self {
            shape: (self.rows(), n_columns),
            values: column_vec,
        }
    }

    pub fn slice_top(&self, n_rows: usize) -> Self {
        assert!(
            n_rows < self.rows(),
            "Cannot slice {} rows from Matrix with shape {:?}",
            n_rows,
            self.shape
        );

        let index_start = 0;
        let index_end = n_rows * self.columns();

        Self {
            shape: (n_rows, self.columns()),
            values: self.values[index_start..index_end].to_vec(),
        }
    }

    pub fn slice_bottom(&self, n_rows: usize) -> Self {
        assert!(
            n_rows < self.rows(),
            "Cannot slice {} rows from Matrix with shape {:?}",
            n_rows,
            self.shape
        );

        let index_start = (self.rows() - n_rows) * self.columns();
        let index_end = self.size();

        Self {
            shape: (n_rows, self.columns()),
            values: self.values[index_start..index_end].to_vec(),
        }
    }
}

impl Clone for Matrix2D {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape,
            values: self.values.clone(),
        }
    }
}

// Operator overloading

macro_rules! impl_add_matrix2d {
    ($lhs:ty, $rhs:ty) => {
        impl Add<$rhs> for $lhs {
            type Output = Matrix2D;

            fn add(self, rhs: $rhs) -> Self::Output {
                assert_eq!(
                    self.shape, rhs.shape,
                    "Cannot add Matrices with different shapes: {:?} != {:?}.",
                    self.shape, rhs.shape
                );

                let mut res = vec![0.0; self.size()];
                for i in 0..self.size() {
                    res[i] = self.values[i] + rhs.values[i];
                }

                Matrix2D::from_vec(self.shape, res)
            }
        }
    };
}

macro_rules! impl_add_assign_matrix2d {
    ($lhs:ty, $rhs:ty) => {
        impl AddAssign<$rhs> for $lhs {
            fn add_assign(&mut self, rhs: $rhs) {
                assert_eq!(
                    self.shape, rhs.shape,
                    "Cannot add Matrices with different shapes: {:?} != {:?}.",
                    self.shape, rhs.shape
                );

                for i in 0..self.size() {
                    self.values[i] += rhs.values[i];
                }
            }
        }
    };
}

macro_rules! impl_sub_matrix2d {
    ($lhs:ty, $rhs:ty) => {
        impl Sub<$rhs> for $lhs {
            type Output = Matrix2D;

            fn sub(self, rhs: $rhs) -> Self::Output {
                assert_eq!(
                    self.shape, rhs.shape,
                    "Cannot subtract Matrices with different shapes: {:?} != {:?}.",
                    self.shape, rhs.shape
                );

                let res = self
                    .get_vec()
                    .iter()
                    .zip(rhs.get_vec())
                    .map(|(lhs, rhs)| lhs - rhs)
                    .collect();

                Matrix2D::from_vec(self.shape, res)
            }
        }
    };
}

macro_rules! impl_sub_assign_matrix2d {
    ($lhs:ty, $rhs:ty) => {
        impl SubAssign<$rhs> for $lhs {
            fn sub_assign(&mut self, rhs: $rhs) {
                assert_eq!(
                    self.shape, rhs.shape,
                    "Cannot subtract Matrices with different shapes: {:?} != {:?}.",
                    self.shape, rhs.shape
                );

                for i in 0..self.size() {
                    self.values[i] -= rhs.values[i];
                }
            }
        }
    };
}

macro_rules! impl_mul_matrix2d {
    ($lhs:ty, $rhs:ty) => {

        impl Mul<$rhs> for $lhs {
            type Output = Matrix2D;

            fn mul(self, rhs: $rhs) -> Self::Output {
                assert_eq!(self.columns(), rhs.rows(), "Cannot multiply Matrices, columns of lhs must be equal to rows of rhs: {:?} * {:?} -> {} != {}", self.shape, rhs.shape, self.columns(), rhs.rows());

                let mut res = Matrix2D::zeros((self.rows(), rhs.columns()));
                for row in 0..res.rows() {
                    for col in 0..res.columns() {
                        let mut dot = 0.0;
                        for i in 0..self.columns() {
                            dot += self[(row, i)] * rhs[(i, col)];
                        }
                        res[(row, col)] = dot;
                    }
                }

                res
            }
        }

    };
}

macro_rules! impl_mul_scalar {
    ($lhs:ty, $rhs:ty) => {
        impl Mul<$rhs> for $lhs {
            type Output = Matrix2D;

            fn mul(self, rhs: $rhs) -> Self::Output {
                let res = self.get_vec().iter().map(|x| x * rhs).collect();
                Matrix2D::from_vec(self.shape, res)
            }
        }

        impl Mul<$lhs> for $rhs {
            type Output = Matrix2D;

            fn mul(self, rhs: $lhs) -> Self::Output {
                rhs.mul(self)
            }
        }
    };
}

macro_rules! impl_mul_assign_matrix2d {
    ($lhs:ty, $rhs:ty) => {
        impl MulAssign<$rhs> for $lhs {
            fn mul_assign(&mut self, rhs: $rhs) {
                *self = self.mul(rhs);
            }
        }
    };
}

macro_rules! impl_div_scalar {
    ($lhs:ty, $rhs:ty) => {
        impl Div<$rhs> for $lhs {
            type Output = Matrix2D;

            fn div(self, rhs: $rhs) -> Self::Output {
                let res = self.get_vec().iter().map(|x| x / rhs).collect();
                Matrix2D::from_vec(self.shape, res)
            }
        }
    };
}

impl_add_matrix2d!(Matrix2D, Matrix2D);
impl_add_matrix2d!(Matrix2D, &Matrix2D);
impl_add_matrix2d!(&Matrix2D, Matrix2D);
impl_add_matrix2d!(&Matrix2D, &Matrix2D);
impl_add_matrix2d!(Matrix2D, &mut Matrix2D);
impl_add_matrix2d!(&mut Matrix2D, Matrix2D);
impl_add_matrix2d!(&mut Matrix2D, &mut Matrix2D);

impl_add_assign_matrix2d!(Matrix2D, Matrix2D);
impl_add_assign_matrix2d!(Matrix2D, &Matrix2D);
impl_add_assign_matrix2d!(Matrix2D, &mut Matrix2D);
// impl_add_assign_matrix2d!(&mut Matrix2D, Matrix2D);
// impl_add_assign_matrix2d!(&mut Matrix2D, &Matrix2D);
// impl_add_assign_matrix2d!(&mut Matrix2D, &mut Matrix2D);

impl_sub_matrix2d!(Matrix2D, Matrix2D);
impl_sub_matrix2d!(Matrix2D, &Matrix2D);
impl_sub_matrix2d!(&Matrix2D, Matrix2D);
impl_sub_matrix2d!(&Matrix2D, &Matrix2D);
impl_sub_matrix2d!(Matrix2D, &mut Matrix2D);
impl_sub_matrix2d!(&mut Matrix2D, Matrix2D);
impl_sub_matrix2d!(&mut Matrix2D, &mut Matrix2D);

impl_sub_assign_matrix2d!(Matrix2D, Matrix2D);
impl_sub_assign_matrix2d!(Matrix2D, &Matrix2D);
impl_sub_assign_matrix2d!(Matrix2D, &mut Matrix2D);
// impl_sub_assign_matrix2d!(&mut Matrix2D, Matrix2D);
// impl_sub_assign_matrix2d!(&mut Matrix2D, &Matrix2D);
// impl_sub_assign_matrix2d!(&mut Matrix2D, &mut Matrix2D);

impl_mul_matrix2d!(Matrix2D, Matrix2D);
impl_mul_matrix2d!(Matrix2D, &Matrix2D);
impl_mul_matrix2d!(Matrix2D, &mut Matrix2D);
impl_mul_matrix2d!(&Matrix2D, Matrix2D);
impl_mul_matrix2d!(&Matrix2D, &Matrix2D);
impl_mul_matrix2d!(&Matrix2D, &mut Matrix2D);
impl_mul_matrix2d!(&mut Matrix2D, Matrix2D);
impl_mul_matrix2d!(&mut Matrix2D, &Matrix2D);
impl_mul_matrix2d!(&mut Matrix2D, &mut Matrix2D);

impl_mul_scalar!(Matrix2D, f32);
impl_mul_scalar!(&Matrix2D, f32);
impl_mul_scalar!(&mut Matrix2D, f32);

impl_mul_assign_matrix2d!(Matrix2D, Matrix2D);
impl_mul_assign_matrix2d!(Matrix2D, &Matrix2D);
impl_mul_assign_matrix2d!(Matrix2D, &mut Matrix2D);

impl_div_scalar!(Matrix2D, f32);
impl_div_scalar!(&Matrix2D, f32);
impl_div_scalar!(&mut Matrix2D, f32);

impl Index<Shape> for Matrix2D {
    type Output = f32;

    fn index(&self, index: Shape) -> &Self::Output {
        assert!(
            index.0 < self.rows() && index.1 < self.columns(),
            "Index out of bounds for Matrix with shape {:?}: {:?}",
            self.shape,
            index
        );
        let i = self.shape_to_vec_index(index);
        &self.values[i]
    }
}

impl IndexMut<Shape> for Matrix2D {
    fn index_mut(&mut self, index: Shape) -> &mut f32 {
        assert!(
            index.0 < self.rows() && index.1 < self.columns(),
            "Index out of bounds for Matrix with shape {:?}: {:?}",
            self.shape,
            index
        );
        let i = self.shape_to_vec_index(index);
        &mut self.values[i]
    }
}

impl MulAssign<f32> for Matrix2D {
    fn mul_assign(&mut self, rhs: f32) {
        for i in 0..self.size() {
            self.values[i] *= rhs;
        }
    }
}

impl DivAssign<f32> for Matrix2D {
    fn div_assign(&mut self, rhs: f32) {
        for i in 0..self.size() {
            self.values[i] /= rhs;
        }
    }
}

impl Debug for Matrix2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\n[")?;
        for r in 0..self.rows() {
            write!(f, "\n    {:?}", self.get_row(r).values)?;
        }
        write!(f, "\n]")
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_size() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = Matrix2D::from_vec((2, 3), v);

        assert_eq!(6, m.size());
    }

    #[test]
    fn test_transpose() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = Matrix2D::from_vec((2, 3), v);

        let m_t = m.transpose();

        let v_t = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];

        assert_eq!(m_t.shape, (3, 2));

        for i in 0..m.size() {
            assert_eq!(m_t.values[i], v_t[i]);
        }
    }

    #[test]
    fn test_reshape() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = Matrix2D::from_vec((2, 3), v);

        let m2 = m.reshape((1, 6));

        assert_eq!(m[(1, 1)], m2[(0, 4)]);
    }

    #[test]
    fn test_reshape_mut() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = Matrix2D::from_vec((2, 3), v);

        let mut m2 = m.clone();
        m2.reshape_mut((1, 6));

        assert_eq!(m[(1, 1)], m2[(0, 4)]);
    }

    #[test]
    fn test_map() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = Matrix2D::from_vec((2, 3), v);

        let m2 = m.map(|x| x * x);

        assert_eq!(m2[(1, 1)], 25.0);
    }

    #[test]
    fn test_map_mut() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut m = Matrix2D::from_vec((2, 3), v);

        m.map_mut(|x| x * x);

        assert_eq!(m[(1, 1)], 25.0);
    }

    #[test]
    fn test_filter() {}

    #[test]
    fn test_filter_mut() {}

    #[test]
    fn test_fold() {}

    #[test]
    fn test_vstack() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = Matrix2D::from_vec((2, 3), v);

        let v2 = vec![7.0, 8.0, 9.0];
        let m2 = Matrix2D::from_vec((1, 3), v2);

        let m3 = Matrix2D::vstack(&m, &m2);

        assert_eq!(m3.shape, (3, 3));

        assert_eq!(m3[(2, 1)], 8.0);
    }

    #[test]
    fn test_hstack() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = Matrix2D::from_vec((2, 3), v);

        let v2 = vec![7.0, 8.0, 9.0, 10.0];
        let m2 = Matrix2D::from_vec((2, 2), v2);

        let m3 = Matrix2D::hstack(&m, &m2);

        assert_eq!(m3.shape, (2, 5));

        assert_eq!(m3[(0, 4)], 8.0);
        assert_eq!(m3[(1, 3)], 9.0);
    }

    #[test]
    fn test_get_row() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = Matrix2D::from_vec((2, 3), v);

        let r = m.get_row(1);

        assert_eq!(r.shape, (1, 3));

        assert_eq!(r[(0, 1)], 5.0);
        assert_eq!(r[(0, 2)], 6.0);
    }

    #[test]
    fn test_get_column() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = Matrix2D::from_vec((2, 3), v);

        let r = m.get_column(1);

        assert_eq!(r.shape, (2, 1));

        assert_eq!(r[(0, 0)], 2.0);
        assert_eq!(r[(1, 0)], 5.0);
    }

    #[test]
    fn test_slice_top() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let m = Matrix2D::from_vec((4, 2), v);

        let m2 = m.slice_top(2);

        assert_eq!(m2.shape, (2, 2));

        assert_eq!(m2[(0, 0)], 1.0);
        assert_eq!(m2[(0, 1)], 2.0);
        assert_eq!(m2[(1, 0)], 3.0);
        assert_eq!(m2[(1, 1)], 4.0);
    }

    #[test]
    fn test_slice_bottom() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let m = Matrix2D::from_vec((4, 2), v);

        let m2 = m.slice_bottom(2);

        assert_eq!(m2.shape, (2, 2));

        assert_eq!(m2[(0, 0)], 5.0);
        assert_eq!(m2[(0, 1)], 6.0);
        assert_eq!(m2[(1, 0)], 7.0);
        assert_eq!(m2[(1, 1)], 8.0);
    }

    #[test]
    fn test_slice_front() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = Matrix2D::from_vec((2, 3), v);

        let m2 = m.slice_front(2);

        assert_eq!(m2.shape, (2, 2));

        assert_eq!(m2[(0, 0)], 1.0);
        assert_eq!(m2[(0, 1)], 2.0);
        assert_eq!(m2[(1, 0)], 4.0);
        assert_eq!(m2[(1, 1)], 5.0);
    }

    #[test]
    fn test_slice_back() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = Matrix2D::from_vec((2, 3), v);

        let m2 = m.slice_back(2);

        assert_eq!(m2.shape, (2, 2));

        assert_eq!(m2[(0, 0)], 2.0);
        assert_eq!(m2[(0, 1)], 3.0);
        assert_eq!(m2[(1, 0)], 5.0);
        assert_eq!(m2[(1, 1)], 6.0);
    }

    #[test]
    fn test_add() {
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m1 = Matrix2D::from_vec((2, 3), v1);

        let v2 = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let m2 = Matrix2D::from_vec((2, 3), v2);

        let m3 = &m1 + &m2;

        assert_eq!(m3.shape, (2, 3));

        assert_eq!(m3[(0, 0)], 1.0 + 7.0);
        assert_eq!(m3[(0, 2)], 3.0 + 9.0);
        assert_eq!(m3[(1, 1)], 5.0 + 11.0);
    }

    #[test]
    fn test_add_assign() {
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut m1 = Matrix2D::from_vec((2, 3), v1);

        let v2 = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let m2 = Matrix2D::from_vec((2, 3), v2);

        m1 += &m2;

        assert_eq!(m1.shape, (2, 3));

        assert_eq!(m1[(0, 0)], 1.0 + 7.0);
        assert_eq!(m1[(0, 2)], 3.0 + 9.0);
        assert_eq!(m1[(1, 1)], 5.0 + 11.0);
    }

    #[test]
    fn test_sub() {
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m1 = Matrix2D::from_vec((2, 3), v1);

        let v2 = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let m2 = Matrix2D::from_vec((2, 3), v2);

        let m3 = &m2 - &m1;

        assert_eq!(m3.shape, (2, 3));

        assert_eq!(m3[(0, 0)], 7.0 - 1.0);
        assert_eq!(m3[(0, 2)], 9.0 - 3.0);
        assert_eq!(m3[(1, 1)], 11.0 - 5.0);
    }

    #[test]
    fn test_sub_assign() {
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m1 = Matrix2D::from_vec((2, 3), v1);

        let v2 = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut m2 = Matrix2D::from_vec((2, 3), v2);

        m2 -= &m1;

        assert_eq!(m1.shape, (2, 3));

        assert_eq!(m2[(0, 0)], 7.0 - 1.0);
        assert_eq!(m2[(0, 2)], 9.0 - 3.0);
        assert_eq!(m2[(1, 1)], 11.0 - 5.0);
    }

    #[test]
    fn test_mul_matrix() {
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m1 = Matrix2D::from_vec((2, 3), v1);

        let m2 = m1.transpose();

        let m3 = &m1 * &m2;

        assert_eq!(m3.shape, (2, 2));

        assert_eq!(m3[(0, 0)], 14.0);
        assert_eq!(m3[(0, 1)], 32.0);
        assert_eq!(m3[(1, 0)], 32.0);
        assert_eq!(m3[(1, 1)], 77.0);
    }

    #[test]
    fn test_mul_assign_matrix() {
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut m1 = Matrix2D::from_vec((2, 3), v1);

        let m2 = m1.transpose();

        m1 *= &m2;

        assert_eq!(m1.shape, (2, 2));

        assert_eq!(m1[(0, 0)], 14.0);
        assert_eq!(m1[(0, 1)], 32.0);
        assert_eq!(m1[(1, 0)], 32.0);
        assert_eq!(m1[(1, 1)], 77.0);
    }

    #[test]
    fn test_mul_f32() {
        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let m1 = Matrix2D::from_vec((2, 2), v1);

        let m2 = &m1 * 2.0;
        let m3 = 2.0 * &m2;

        assert_eq!(m3.shape, (2, 2));

        assert_eq!(m3[(0, 0)], 4.0);
        assert_eq!(m3[(0, 1)], 8.0);
        assert_eq!(m3[(1, 0)], 12.0);
        assert_eq!(m3[(1, 1)], 16.0);
    }

    #[test]
    fn test_mul_assign_f32() {
        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let mut m1 = Matrix2D::from_vec((2, 2), v1);

        m1 *= 2.0;

        assert_eq!(m1.shape, (2, 2));

        assert_eq!(m1[(0, 0)], 2.0);
        assert_eq!(m1[(0, 1)], 4.0);
        assert_eq!(m1[(1, 0)], 6.0);
        assert_eq!(m1[(1, 1)], 8.0);
    }

    #[test]
    fn test_div_f32() {
        let v1 = vec![2.0, 4.0, 6.0, 8.0];
        let m1 = Matrix2D::from_vec((2, 2), v1);

        let m2 = &m1 / 2.0;

        assert_eq!(m2.shape, (2, 2));

        assert_eq!(m2[(0, 0)], 1.0);
        assert_eq!(m2[(0, 1)], 2.0);
        assert_eq!(m2[(1, 0)], 3.0);
        assert_eq!(m2[(1, 1)], 4.0);
    }

    #[test]
    fn test_div_assign_f32() {
        let v1 = vec![2.0, 4.0, 6.0, 8.0];
        let mut m1 = Matrix2D::from_vec((2, 2), v1);

        m1 /= 2.0;

        assert_eq!(m1.shape, (2, 2));

        assert_eq!(m1[(0, 0)], 1.0);
        assert_eq!(m1[(0, 1)], 2.0);
        assert_eq!(m1[(1, 0)], 3.0);
        assert_eq!(m1[(1, 1)], 4.0);
    }
}
