use num_traits::Zero;

use crate::field::F128;

pub trait LinOp {
    fn n_in(&self) -> usize;
    fn n_out(&self) -> usize;
    
    /// expects input of size n_in and output of size n_out
    fn apply(&self, input: &[F128], output: &mut [F128]);
        /// expects input of size n_out and output of size n_in
    fn apply_transposed(&self, input: &[F128], output: &mut [F128]);
}

pub struct Composition <
    A: LinOp,
    B: LinOp
> {
    a: A,
    b: B,
}

impl<
    A: LinOp,
    B: LinOp
>
Composition<A, B> {
    pub fn new(a: A, b: B) -> Self {
        assert!(b.n_out() == a.n_in());
        Self { a, b }
    }
}

impl<
    A: LinOp,
    B: LinOp
>
LinOp for Composition<A, B> {
        
    fn n_in(&self) -> usize {
        self.b.n_in()
    }
    
    fn n_out(&self) -> usize {
        self.a.n_out()
    }

    fn apply(&self, input: &[F128], output: &mut [F128]) {
        let mid = self.b.n_out();
        let mut tmp = vec![F128::zero(); mid];
        self.b.apply(input, &mut tmp);
        self.a.apply(&tmp, output);
    }

    fn apply_transposed(&self, input: &[F128], output: &mut [F128]) {
        let mid = self.b.n_out();
        let mut tmp = vec![F128::zero(); mid];
        self.a.apply_transposed(input, &mut tmp);
        self.b.apply_transposed(&tmp, output);
    }

}