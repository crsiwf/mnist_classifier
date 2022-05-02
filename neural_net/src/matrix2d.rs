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
        self.fold(f32::INFINITY, f32::min)
    }

    pub fn argmin(&self) -> Shape {
        let mut x = f32::INFINITY;
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


    pub fn hslice(&self, slice_at: usize) -> (Self, Self) {
        assert!(
            slice_at < self.columns() + 1,
            "Cannot slice Matrix with shape {:?} at column {}",
            self.shape,
            slice_at,
        );

        let front_size = slice_at * self.rows();
        let mut vec_front = vec![0.0; front_size];
        let mut vec_back = vec![0.0; self.size() - front_size];

        println!("{} {}", vec_front.len(), vec_back.len());

        for r in 0..self.rows() {
            for c in 0..slice_at {
                vec_front[r * slice_at + c] = self[(r, c)];
            }
            for c in slice_at..self.columns() {
                vec_back[r * (self.columns() - slice_at) + c - slice_at] = self[(r, c)];
            }
        }

        let front = Self {
            shape: (self.rows(), slice_at),
            values: vec_front,
        };
        let back = Self {
            shape: (self.rows(), self.columns() - slice_at),
            values: vec_back,
        };

        (front, back)
    }

    pub fn vslice(&self, slice_at: usize) -> (Self, Self) {
        assert!(
            slice_at < self.rows() + 1,
            "Cannot slice Matrix with shape {:?} at row {}",
            self.shape,
            slice_at,
        );

        let slice_index = slice_at * self.columns();

        let top = Self {
            shape: (slice_at, self.columns()),
            values: self.values[..slice_index].to_vec(),
        };

        let bottom = Self {
            shape: (self.rows() - slice_at, self.columns()),
            values: self.values[slice_index..].to_vec(),
        };

        (top, bottom)
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

impl Debug for Matrix2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\n[")?;
        for r in 0..self.rows() {
            write!(f, "\n    {:?}", self.get_row(r).values)?;
        }
        write!(f, "\n]")
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