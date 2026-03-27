/**
 * @file spectral.h
 * @brief Chiral spectral preprocessing for Kissat SAT solver.
 *
 * Computes variable polarity hints from a chiral multi-shell spectral
 * decomposition of the clause-variable graph. The Fiedler vector of
 * the metallic-mean-weighted Laplacian provides an informed initial
 * partition that reduces the conflict count for hard instances.
 *
 * Integration: called once after parsing, before CDCL search.
 * Sets solver->phases.target[idx] for all variables.
 * Zero changes to the CDCL core.
 *
 * Algorithm: AntiResonantSAT (Knopp, 2026)
 * - Chiral edge weighting: cos(omega*dt) + chi*sin(omega*dt)
 * - Metallic-mean phase spacing: theta_k = 2*pi * beta^k / sum(beta^j)
 * - Three shells: Bronze(-1), Silver(+1), Golden(-1)
 * - Compound voting across shells
 *
 * Reference: "Chiral Spectral Heuristics for Max-SAT" (2026)
 *
 * @author Christian Knopp / Zynerji Research
 * @license MIT (same as Kissat)
 */

#ifndef _spectral_h_INCLUDED
#define _spectral_h_INCLUDED

struct kissat;

/**
 * Apply chiral spectral preprocessing.
 *
 * Builds a metallic-mean-weighted Laplacian from the clause-variable
 * graph, computes the Fiedler vector via shifted power iteration, and
 * sets variable target phases accordingly.
 *
 * Runs in O(iterations * edges) time, typically ~50ms for n=1000.
 * Called once, before kissat_solve().
 *
 * @param solver The Kissat solver instance (after parsing).
 */
void kissat_spectral_preprocessing (struct kissat *solver);

#endif
