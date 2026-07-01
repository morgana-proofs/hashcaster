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
use crate::{examples::keccak::{chi_round::{chi_round_witness_into, ChiPackage}, matrices::{keccak_linround_witness_into, KeccakLinMatrix}}, field::F128, protocols::{boolcheck::{BoolCheck, BoolCheckOutput, FnPackage}, lincheck::{LinOp, Lincheck, LincheckOutput}, multiclaim::MulticlaimCheck, utils::{eq_ev, eq_poly, evaluate, evaluate_univar, untwist_evals}}, traits::SumcheckObject};

#[test]
pub fn main_protocol() {

    println!("... Generating initial randomness, it might take some time ...");

    let rng = &mut OsRng;
    let num_vars = 20;
    let c = 5;
    let num_active_vars = 10;

    let pt : Vec<F128> = (0..num_vars).map(|_| F128::rand(rng)).collect();

    let mut polys : Vec<Vec<F128>> = vec![];
    for _ in 0..5 {
        polys.push((0 .. 1 << num_vars).map(|_| F128::rand(rng)).collect());
    }

    println!("... Preparing witness...");

    let mut eq_scratch = vec![F128::zero(); 1 << pt.len()];

    let layer0 : [Vec<F128>; 5] = polys.try_into().unwrap();
    let polys_refs = layer0.iter().map(|x| x.as_slice()).collect::<Vec<_>>().try_into().unwrap();
    let mut layer1 : [Vec<F128>; 5] = (0..5).map(|_| vec![F128::zero(); 1 << num_vars]).collect::<Vec<_>>().try_into().unwrap();
    let mut layer2 : [Vec<F128>; 5] = (0..5).map(|_| vec![F128::zero(); 1 << num_vars]).collect::<Vec<_>>().try_into().unwrap();

    let boolcheck_pow3 = 3usize.pow((c + 1) as u32);
    let boolcheck_pow2 = 1 << (num_vars - c - 1);
    let boolcheck_pow3_adj = boolcheck_pow3 / 3 * 2;
    let boolcheck_ext = vec![F128::zero(); boolcheck_pow3 * boolcheck_pow2];
    let boolcheck_ext_scratch = vec![F128::zero(); boolcheck_pow3 * boolcheck_pow2];
    #[cfg(feature = "parallel")]
    let boolcheck_table_ext_copies = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let boolcheck_table_ext_copies = 1;
    let boolcheck_tables_ext : Vec<Vec<F128>> = (0..5 * boolcheck_table_ext_copies).map(|_| vec![F128::zero(); boolcheck_pow3_adj]).collect();
    let boolcheck_tail_eq_low = vec![F128::zero(); 1 << ((num_vars + 1) / 2)];
    let boolcheck_tail_eq_high = vec![F128::zero(); (1 << ((num_vars + 1) / 2)).max(1 << (num_vars - c - 1))];
    let boolcheck_poly_coords = vec![F128::zero(); 5 * 128 * (1 << (num_vars - c - 1))];
    let boolcheck_restrict_eq = vec![F128::zero(); 1 << (c + 1)];
    let boolcheck_restrict_eq_sums = vec![F128::zero(); 256 * boolcheck_restrict_eq.len() / 8];
    let boolcheck_pt = pt.clone();
    let mut boolcheck_rs = Vec::with_capacity(num_vars);
    let mut coord_evals = vec![F128::zero(); 128 * 5 + 1];

    let multi_poly = vec![F128::zero(); 1 << num_vars];
    let multi_eq = vec![F128::zero(); 1 << num_vars];
    let multi_p_scratch = vec![vec![F128::zero(); 1 << num_vars]; 1];
    let multi_q_scratch = vec![vec![F128::zero(); 1 << num_vars]; 1];
    let mut pt_inv_orbit : Vec<Vec<F128>> = (0..128).map(|_| vec![F128::zero(); num_vars]).collect();
    let mut tmp_orbit_pt = Vec::with_capacity(num_vars);
    let mut multiopen_rs = Vec::with_capacity(num_vars);

    let lin_chunk = 1 << num_active_vars;
    let lin_eq_dormant = vec![F128::zero(); 1 << (num_vars - num_active_vars)];
    let lin_p_polys = vec![vec![F128::zero(); lin_chunk]; 5];
    let lin_eq = vec![F128::zero(); lin_chunk];
    let lin_gamma_eqs = vec![F128::zero(); 5 * lin_chunk];
    let lin_q = vec![F128::zero(); 5 * lin_chunk];
    let lin_q_polys = vec![vec![F128::zero(); lin_chunk]; 5];
    let lin_p_scratch = vec![vec![F128::zero(); lin_chunk]; 5];
    let lin_q_scratch = vec![vec![F128::zero(); lin_chunk]; 5];
    let mut lin_rs = Vec::with_capacity(num_active_vars);
    let mut lin_eq1 = vec![F128::zero(); lin_chunk];
    let mut lin_eq0 = vec![F128::zero(); lin_chunk];
    let mut lin_adj_eq_vec = vec![F128::zero(); 5 * lin_chunk];
    let mut lin_target = vec![F128::zero(); 5 * lin_chunk];
    let mut final_rs = Vec::with_capacity(num_vars);

    let wtns_start = Instant::now();

    keccak_linround_witness_into(polys_refs, &mut layer1);
    chi_round_witness_into(&layer1, &mut layer2);

    let wtns_finish = Instant::now();

    println!(">>>> Witness gen took {} ms", (wtns_finish - wtns_start).as_millis());

    let evaluation_claims : [F128; 5] = std::array::from_fn(|i| evaluate(&layer2[i], &pt, &mut eq_scratch));

    let evaluations_finish = Instant::now();

    println!(">>>> Evaluation of output took {} ms", (evaluations_finish - wtns_finish).as_millis());

    let f = ChiPackage{};

    println!(">> Total witness / claim generation time: {} ms", (evaluations_finish - wtns_start).as_millis());

    // ------------------ Boolcheck layer ---------------------

    let boolcheck_start = Instant::now();

    let prover = BoolCheck::new(
        f,
        &layer1, 
        c,
        evaluation_claims,
        boolcheck_pt
    );

    let boolcheck_init = Instant::now();

    println!(">>>> Initialization took: {} ms", (boolcheck_init - boolcheck_start).as_millis());

    let gamma = F128::rand(rng);
    let mut prover = prover.folding_challenge(gamma, boolcheck_ext, boolcheck_ext_scratch, boolcheck_tables_ext, boolcheck_tail_eq_low, boolcheck_tail_eq_high, boolcheck_poly_coords, boolcheck_restrict_eq, boolcheck_restrict_eq_sums);

    let boolcheck_extensions = Instant::now();

    println!(">>>> Table extension took: {} ms", (boolcheck_extensions - boolcheck_init).as_millis());

    // Initialize expected (folded) claim.
    let mut claim = evaluate_univar(&evaluation_claims, gamma);

    boolcheck_rs.clear();

    for i in 0..num_vars {
        let rpoly = prover.round_msg().coeffs(claim);

        let r = F128::rand(rng);
        assert!(rpoly.len() == 4);
        claim = evaluate_univar(&rpoly, r);
        prover.bind(r);
        boolcheck_rs.push(r);
    }

    let BoolCheckOutput { frob_evals, .. } = prover.finish();

    let boolcheck_final = Instant::now();

    println!(">>>> Rounds took: {} ms", (boolcheck_final - boolcheck_extensions).as_millis());

    assert!(frob_evals.len() == 128 * 5);

    coord_evals[..frob_evals.len()].copy_from_slice(&frob_evals);
    coord_evals[..frob_evals.len()].chunks_mut(128).map(|chunk| untwist_evals(chunk)).count();

    coord_evals[frob_evals.len()] = F128::zero(); // Ugly hack.
    let claimed_ev = ChiPackage{}.exec_alg(&coord_evals[..frob_evals.len() + 1], 0, 1)[0];
    
    let folded_claimed_ev = evaluate_univar(&claimed_ev, gamma);
    assert!(folded_claimed_ev * eq_ev(&pt, &boolcheck_rs) == claim); // Boolcheck final check.

    let boolcheck_final_verify = Instant::now();

    println!(">>>> Verifier took: {} ms", (boolcheck_final_verify - boolcheck_final).as_millis());

    println!(">> Boolcheck total time: {} ms", (boolcheck_final_verify - boolcheck_start).as_millis());
    // ----------- Multiopen layer --------------

    println!("... Entering multiopen phase ...");

    let pt = &boolcheck_rs;

    let multiopen_start = Instant::now();

    tmp_orbit_pt.clear();
    tmp_orbit_pt.extend_from_slice(pt);
    for orbit_idx in (0..128).rev() {
        tmp_orbit_pt.iter_mut().map(|x| *x *= *x).count();
        pt_inv_orbit[orbit_idx].copy_from_slice(&tmp_orbit_pt);
    }

    let prover = MulticlaimCheck::new(&layer1, pt, &frob_evals);
    
    let gamma = F128::rand(rng);
    
    let mut gamma128 = gamma;
    for i in 0..7 {
        gamma128 *= gamma128;
    }

    let mut prover = prover.folding_challenge(gamma, multi_poly, multi_eq, multi_p_scratch, multi_q_scratch);
    
    let mut claim = evaluate_univar(&frob_evals, gamma);
    multiopen_rs.clear();
    for i in 0..num_vars {
        let round_poly = prover.round_msg();
        let r = F128::rand(rng);
        multiopen_rs.push(r);
        let decomp_rpoly = round_poly.coeffs(claim);
        claim = 
            decomp_rpoly[0] + r * decomp_rpoly[1] + r * r * decomp_rpoly[2];
        prover.bind(r);
    }

    let evals = prover.finish();

    let eq_evs : [F128; 128] = std::array::from_fn(|i| eq_ev(&pt_inv_orbit[i], &multiopen_rs));

    let eq_ev = evaluate_univar(&eq_evs, gamma);
    let eval = evaluate_univar(&evals, gamma128);

    assert!(eval * eq_ev == claim); // Multiopen final check.

    let multiopen_end = Instant::now();

    println!(">> Multiopen took {} ms", (multiopen_end - multiopen_start).as_millis());

    // --------------- Linear layer ---------------

    println!("... Entering linear layer ...");

    let pt = &multiopen_rs;
    let matrix = KeccakLinMatrix::new();
    let evals : [F128; 5] = evals.try_into().unwrap();

    let linlayer_start = Instant::now();

    let prover = Lincheck::new(&layer0, pt, matrix, num_active_vars, evals);

    let gamma = F128::rand(rng);
    let mut prover = prover.folding_challenge(gamma, lin_eq_dormant, lin_p_polys, lin_eq, lin_gamma_eqs, lin_q, lin_q_polys, lin_p_scratch, lin_q_scratch);
    let mut claim = evaluate_univar(&evals, gamma);

    let linlayer_clone_restrict = Instant::now();

    println!(">>>> Data prep (clone/restrict) took {} ms", (linlayer_clone_restrict - linlayer_start).as_millis());

    lin_rs.clear();
    for _ in 0..num_active_vars {
        let rpoly = prover.round_msg().coeffs(claim);
        let r = F128::rand(rng);
        claim = rpoly[0] + rpoly[1] * r + rpoly[2] * r * r;
        prover.bind(r);
        lin_rs.push(r);
    };


    let LincheckOutput {p_evs: l0_evals, ..} = prover.finish();

    assert!(l0_evals.len() == 5);

    eq_poly(&pt[..num_active_vars], &mut lin_eq1);
    eq_poly(&lin_rs, &mut lin_eq0);

    let mut mult = F128::one();
    for i in 0..5 {
        for j in 0..lin_chunk {
            lin_adj_eq_vec[i * lin_chunk + j] = lin_eq1[j] * mult;
        }
        mult *= gamma;
    }
    let m = KeccakLinMatrix::new();

    lin_target.fill(F128::zero());
    m.apply_transposed(&lin_adj_eq_vec, &mut lin_target);

    let mut eq_evals = [F128::zero(); 5];
    for i in 0..5 {
        eq_evals[i] =
            lin_target[i * lin_chunk .. (i + 1) * lin_chunk].iter()
                .zip(lin_eq0.iter())
                .map(|(a, b)| *a * b)
                .fold(F128::zero(), |a, b| a + b);
    }

    let expected_claim = l0_evals.iter()
        .zip_eq(eq_evals.iter())
        .map(|(a, b)| *a * b)
        .fold(F128::zero(), |a, b| a + b);

    assert!(expected_claim == claim); // Final check of linear layer.

    let linlayer_end = Instant::now();

    final_rs.clear();
    final_rs.extend_from_slice(&lin_rs);
    final_rs.extend(pt[num_active_vars..].iter().map(|x| *x));
    assert!(final_rs.len() == num_vars);

    for i in 0..5 {
        assert!(evaluate(&layer0[i], &final_rs, &mut eq_scratch) == l0_evals[i]);
    }
    println!(">>>> Main cycle took {} ms", (linlayer_end - linlayer_clone_restrict).as_millis());

    println!(">> Linlayer took {} ms", (linlayer_end - linlayer_start).as_millis());

    println!("TOTAL TIME: {} ms", (linlayer_end - wtns_start).as_millis());
}
