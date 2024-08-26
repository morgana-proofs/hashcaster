// This implements an (interactive, only useful for testing) keccak prover.

// Currently, it is not end-to-end, with both commitment and round wiring lacking. I do not expect it to take more than
// 10-15% of prover time, though, so this is a good first estimate.

// Protocol consists of 3 sumchecks, applied sequentially:

// Boolcheck applied to chi_round, then Multiopen to reduce openings in frobenius orbit to a single opening, and then
// lincheck to apply linear rounds.

// In end-to-end example, the outputs must be wired into inputs (say, using rotation polynomial).

use std::time::Instant;

use itertools::Itertools;
use num_traits::{One, Zero};
use rand::rngs::OsRng;
use crate::{examples::keccak::{chi_round::{chi_round_witness, ChiPackage}, matrices::{keccak_linround_witness, KeccakLinMatrix}}, field::F128, protocols::{boolcheck::{BoolCheck, BoolCheckOutput, FnPackage}, lincheck::{LinOp, Lincheck, LincheckOutput}, multiclaim::MulticlaimCheck, utils::{eq_ev, eq_poly, evaluate, evaluate_univar, untwist_evals}}, traits::SumcheckObject};

#[test]
pub fn main_protocol() {

    println!("... Generating initial randomness, it might take some time ...");

    let rng = &mut OsRng;
    let num_vars = 20;
    let c = 5;

    let pt : Vec<F128> = (0..num_vars).map(|_| F128::rand(rng)).collect();

    let mut polys : Vec<Vec<F128>> = vec![];
    for _ in 0..5 {
        polys.push((0 .. 1 << num_vars).map(|_| F128::rand(rng)).collect());
    }

    println!("... Preparing witness...");

    let wtns_start = Instant::now();

    let layer0 : [Vec<F128>; 5] = polys.try_into().unwrap();
    let polys_refs = layer0.iter().map(|x| x.as_slice()).collect::<Vec<_>>().try_into().unwrap();

    let layer1 = keccak_linround_witness(polys_refs);
    let layer2 = chi_round_witness(&layer1);

    let wtns_finish = Instant::now();

    println!(">>>> Witness gen took {} ms", (wtns_finish - wtns_start).as_millis());

    let evaluation_claims : [F128; 5] = layer2.iter().map(|poly| evaluate(&poly, &pt)).collect::<Vec<F128>>().try_into().unwrap();

    let evaluations_finish = Instant::now();

    println!(">>>> Evaluation of output took {} ms", (evaluations_finish - wtns_finish).as_millis());

    let f = ChiPackage{};

    println!(">> Total witness / claim generation time: {} ms", (evaluations_finish - wtns_start).as_millis());

    // ------------------ Boolcheck layer ---------------------

    let boolcheck_start = Instant::now();

    let prover = BoolCheck::new(
        f,
        layer1.clone(), 
        c,
        evaluation_claims,
        pt.clone()
    );

    let boolcheck_init = Instant::now();

    println!(">>>> Initialization (cloning) took: {} ms", (boolcheck_init - boolcheck_start).as_millis());

    let gamma = F128::rand(rng);
    let mut prover = prover.folding_challenge(gamma);

    let boolcheck_extensions = Instant::now();

    println!(">>>> Table extension took: {} ms", (boolcheck_extensions - boolcheck_init).as_millis());

    // Initialize expected (folded) claim.
    let mut claim = evaluate_univar(&evaluation_claims, gamma);

    let mut rs = vec![];

    for i in 0..num_vars {
        let rpoly = prover.round_msg().coeffs(claim);

        let r = F128::rand(rng);
        assert!(rpoly.len() == 4);
        claim = evaluate_univar(&rpoly, r);
        prover.bind(r);
        rs.push(r);
    }

    let BoolCheckOutput { frob_evals, .. } = prover.finish();

    let boolcheck_final = Instant::now();

    println!(">>>> Rounds took: {} ms", (boolcheck_final - boolcheck_extensions).as_millis());

    assert!(frob_evals.len() == 128 * 5);

    let mut coord_evals = frob_evals.clone();
    coord_evals.chunks_mut(128).map(|chunk| untwist_evals(chunk)).count();

    coord_evals.push(F128::zero()); // Ugly hack.
    let claimed_ev = ChiPackage{}.exec_alg(&coord_evals, 0, 1)[0];
    
    let folded_claimed_ev = evaluate_univar(&claimed_ev, gamma);
    assert!(folded_claimed_ev * eq_ev(&pt, &rs) == claim); // Boolcheck final check.

    let boolcheck_final_verify = Instant::now();

    println!(">>>> Verifier took: {} ms", (boolcheck_final_verify - boolcheck_final).as_millis());

    println!(">> Boolcheck total time: {} ms", (boolcheck_final_verify - boolcheck_start).as_millis());
    // ----------- Multiopen layer --------------

    println!("... Entering multiopen phase ...");

    let multiopen_start = Instant::now();

    let pt = rs;
    let mut pt_inv_orbit = vec![];

    let mut tmp = pt.clone();
    for _ in 0..128 {
        tmp.iter_mut().map(|x| *x *= *x).count();
        pt_inv_orbit.push(
            tmp.clone()
        )
    }
    pt_inv_orbit.reverse();

    let prover = MulticlaimCheck::new(&layer1, pt.clone(), frob_evals.clone());
    
    let gamma = F128::rand(rng);
    
    let mut gamma128 = gamma;
    for i in 0..7 {
        gamma128 *= gamma128;
    }

    let mut prover = prover.folding_challenge(gamma);
    
    let mut claim = evaluate_univar(&frob_evals, gamma);
    let mut rs = vec![];
    for i in 0..num_vars {
        let round_poly = prover.round_msg();
        let r = F128::rand(rng);
        rs.push(r);
        let decomp_rpoly = round_poly.coeffs(claim);
        claim = 
            decomp_rpoly[0] + r * decomp_rpoly[1] + r * r * decomp_rpoly[2];
        prover.bind(r);
    }

    let evals = prover.finish();

    let eq_evs : Vec<_> = pt_inv_orbit.iter()
        .map(|pt| eq_ev(&pt, &rs))
        .collect();

    let eq_ev = evaluate_univar(&eq_evs, gamma);
    let eval = evaluate_univar(&evals, gamma128);

    assert!(eval * eq_ev == claim); // Multiopen final check.

    let multiopen_end = Instant::now();

    println!(">> Multiopen took {} ms", (multiopen_end - multiopen_start).as_millis());

    // --------------- Linear layer ---------------

    println!("... Entering linear layer ...");

    let linlayer_start = Instant::now();

    let pt = rs;
    let matrix = KeccakLinMatrix::new();
    let evals : [F128; 5] = evals.try_into().unwrap();

    let num_active_vars = 10;

    let prover = Lincheck::new(layer0.clone(), pt.clone(), matrix, num_active_vars, evals);

    let gamma = F128::rand(rng);
    let mut prover = prover.folding_challenge(gamma);
    let mut claim = evaluate_univar(&evals, gamma);

    let linlayer_clone_restrict = Instant::now();

    println!(">>>> Data prep (clone/restrict) took {} ms", (linlayer_clone_restrict - linlayer_start).as_millis());

    let mut rs = vec![];
    for _ in 0..num_active_vars {
        let rpoly = prover.round_msg().coeffs(claim);
        let r = F128::rand(rng);
        claim = rpoly[0] + rpoly[1] * r + rpoly[2] * r * r;
        prover.bind(r);
        rs.push(r);
    };


    let LincheckOutput {p_evs: l0_evals, ..} = prover.finish();

    assert!(l0_evals.len() == 5);

    let eq1 = eq_poly(&pt[..num_active_vars]);
    let eq0 = eq_poly(&rs);
    let mut adj_eq_vec = vec![];

    let mut mult = F128::one();
    for i in 0..5 {
        adj_eq_vec.extend(eq1.iter().map(|x| *x * mult));
        mult *= gamma;
    }
    let m = KeccakLinMatrix::new();

    let mut target = vec![F128::zero(); 5 * (1 << num_active_vars)];
    m.apply_transposed(&adj_eq_vec, &mut target);

    let mut eq_evals = vec![];
    for i in 0..5 {
        eq_evals.push(
            target[i * (1 << num_active_vars) .. (i + 1) * (1 << num_active_vars)].iter()
                .zip(eq0.iter())
                .map(|(a, b)| *a * b)
                .fold(F128::zero(), |a, b| a + b));
    }

    let expected_claim = l0_evals.iter()
        .zip_eq(eq_evals.iter())
        .map(|(a, b)| *a * b)
        .fold(F128::zero(), |a, b| a + b);

    assert!(expected_claim == claim); // Final check of linear layer.

    let linlayer_end = Instant::now();

    rs.extend(pt[num_active_vars..].iter().map(|x| *x));

    for i in 0..5 {
        assert!(evaluate(&layer0[i], &rs) == l0_evals[i]);
    }
    println!(">>>> Main cycle took {} ms", (linlayer_end - linlayer_clone_restrict).as_millis());

    println!(">> Linlayer took {} ms", (linlayer_end - linlayer_start).as_millis());

    println!("TOTAL TIME: {} ms", (linlayer_end - wtns_start).as_millis());
}