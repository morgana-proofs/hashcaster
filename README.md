# Hashcaster

This repository contains demonstration of technique of batched small-field sumcheck using Frobenius twists. My mathematical notes are currently in disarray, but I will try to provide extensive commentary.
I hope that following the code you will be able to figure the rest out, and article will be published when I catch my breath.

For now, you can check article stub here: https://hackmd.io/@levs57/SJ4fuZMD0

## Installation instructions

Clone repository, switch to nightly, and run

```
RUSTFLAGS="-Ctarget-cpu=native" RUST_BACKTRACE=1 cargo test main_protocol --release -- --nocapture
```

I have 2 backends - one using AVX-256 instructions, and another using Neon for ARM architecture. There is currently no soft backend, so in case your device does not support any of those, the code will be unsound (and will likely crash).
You will need L1 Cache of size at least 128Kb, or the performance will be suboptimal.

## Acknowledgment

I would like to thank

* My collaborator Olena @Fitznik from Matter Labs for help with various parts of this code and Neon backend
* My colleague Amir @aliventer who taught me how to use Linux Perf and check cache performance
* Shahar Papini @spapinistarkware for reviewing earlier version of this code
* Justin Thaler and Srinath Setty for various discussions about sumcheck

## How it works

### Frobenius unpacking

First, we present a technique comparable to Binius code switch unpacking. It is conceptually easier, allegedly somewhat faster, and it works over any field and in any basis (i.e. it does not require an extension tower).
Let us assume we are in extension $\mathbb{F}^d$ of some base field $\mathbb{F}$, and pick some basis $b_0, ..., b_{d-1}$. Then, there is a packing map from tuples of size $d$, which sends $(x_0, ..., x_{d-1}) \mapsto \underset{i}\sum b_i x_i$.

Now, assume that we have a packed polynomial $P(x) = \underset{i}\sum b_i P_i(x)$. I claim that it is possible to express the coordinates $P_i(x)$ using Frobenius twists $(Fr^{i}P)(x)$ (this is a local fact from linear algebra, and is equivalent to the non-degeneracy of the trace form).

However, Frobenius action on $P$ is related to openings of $P$ in different points - namely, $(Fr^i(P))(x) = Fr^i(P(Fr^{-i}(x)))$. Therefore, we have reduced openings of $P_i$ in a single point $r$ to multiple openings of a polynomial $P$ in an inverse Frobenius orbit of a point $r$.

### Efficient multiopen reduction

Opening polynomial in a Frobenius orbit is much cheaper than in a random set of points. For the reference, check multiclaim.rs file. High-level idea is that for a sum $\underset{i}\sum \gamma_i P(x) eq(Fr^{-i}(r), x)$ it is possible to gather all eq-polynomials and compute it more efficiently by applying the matrix $\underset{i}\sum \gamma_i Fr^{-i}(r)$ to a polynomial $eq(r, x)$.
We use method of 4 Russians to precompute this matrix and then apply it in a very efficient fashion.

### Linear operations

Linear operations in sumcheck are essentially free, provided that they are vectorized enough. Reason for this is that we can reduce the problem to smaller sumcheck:

$$\underset{x_{\text{active}}} \sum L(x_{\text{active}}, r_{\text{active}}) P(x_{\text{active}}, r_{\text{dormant}})$$

only over subset of variables.

### Boolcheck

The core of the argument is boolcheck - for arbitrary quadratic formula over base field we treat it as a formula depending on $P_i$-s - coordinates of our polynomials. There is a novel technique, somewhat inspired by Angus Gruen's univariate skip and recent Justin Thaler's work on small-field sumcheck:

namely, we find an evaluation set $(0, 1, \infty)^{c+1} \times (0, 1)^{n-c-1}$, and extend our polynomials to this larger set. Because it is defined over the base field ($\mathbb{F}_2$), the quadratic formula can be directly applied without passing to the individual coordinates.
This is referred as "table extension", and allows us to pass first $c$ rounds.
https://github.com/morgana-proofs/hashcaster/blob/9e5a35bab73259eb25cec1c72852be46a223f618/src/protocols/utils.rs#L340

Then, we restrict all coordinate polynomials on first $c$ challenges (this is referred as "restrict" and is second heaviest procedure, and it involves a lot of vectorization and another instance of method of 4 Russians), and use algebraic form of the sumcheck to deal with the remaining rounds.
https://github.com/morgana-proofs/hashcaster/blob/9e5a35bab73259eb25cec1c72852be46a223f618/src/protocols/utils.rs#L449

## Performance

These are results from my relatively old 4 core i5:

```
... Generating initial randomness, it might take some time ...
... Preparing witness...
>>>> Witness gen took 114 ms
>>>> Evaluation of output took 40 ms
>> Total witness / claim generation time: 154 ms
>>>> Initialization (cloning) took: 46 ms
>>>> Table extension took: 307 ms
>>>> Rounds took: 483 ms
>>>> Verifier took: 1 ms
>> Boolcheck total time: 838 ms
... Entering multiopen phase ...
>> Multiopen took 74 ms
... Entering linear layer ...
>>>> Data prep (clone/restrict) took 70 ms
>>>> Main cycle took 1 ms
>> Linlayer took 71 ms
TOTAL TIME: 1139 ms
test examples::keccak::main_protocol::main_protocol ... ok
```

be warned that is not end to end test - it lacks both commitment (likely no more than 10-15% of time), and rotation argument to fit inputs with outputs. On other hand, there are known optimizations that I didn't do yet.

## Please help! ^_^

Marketplace of ideas is very inefficient, this is why I've spent a lot of my time to prove that this approach is feasible. But I'm not a cryptographic engineer, I'm a mathematician. I have very poor idea about low level optimizations. My hope is that this gets significant interest from the community and we will get performance to much higher level (while this is already respectable, I believe real engineers could do much, much better).
