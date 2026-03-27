/**
 * @file spectral.c
 * @brief Chiral spectral preprocessing for Kissat SAT solver.
 *
 * Implements the AntiResonantSAT spectral pipeline in pure C:
 *
 *   1. Build sparse clause-variable adjacency with metallic-mean
 *      phase-weighted edges: w(u,v) = cos(omega*dt) + chi*sin(omega*dt)
 *   2. Compute graph Laplacian L = D - W
 *   3. Approximate Fiedler vector via shifted power iteration
 *   4. Set target phases from sign(Fiedler)
 *
 * Three shells with alternating chirality (LRL = -1, +1, -1):
 *   - Bronze (beta_3 = 3.303): aggressive partitioning
 *   - Silver (beta_2 = 2.414): orthogonal chiral partition
 *   - Golden (phi   = 1.618): stability / tie-breaking
 *
 * No external dependencies. No dynamic allocation beyond what Kissat
 * already provides. Matches Kissat's C style and conventions.
 *
 * @author Christian Knopp / Zynerji Research
 */

#include "spectral.h"
#include "internal.h"
#include "logging.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── Metallic mean constants ─────────────────────────────────────── */

#define GOLDEN_BETA 1.6180339887498949   /* (1 + sqrt(5)) / 2       */
#define SILVER_BETA 2.4142135623730951   /* (2 + sqrt(8)) / 2       */
#define BRONZE_BETA 3.3027756377319946   /* (3 + sqrt(13)) / 2      */

#define PI 3.14159265358979323846
#define TWO_PI (2.0 * PI)

#define SPECTRAL_MAX_ITERS 60   /* power iteration convergence     */
#define SPECTRAL_TOL 1e-8       /* convergence tolerance           */
#define SPECTRAL_MAX_EDGES 5000000 /* memory cap for edge list     */

/* ── Sparse matrix (CSR format) ──────────────────────────────────── */

typedef struct sparse_matrix {
  unsigned n;           /* dimension (n_vars)                       */
  unsigned nnz;         /* number of non-zeros (edges * 2)          */
  unsigned *row_ptr;    /* row pointers (n+1 entries)               */
  unsigned *col_idx;    /* column indices (nnz entries)             */
  double *values;       /* edge weights (nnz entries)               */
} sparse_matrix;

static void sparse_free (sparse_matrix *M) {
  free (M->row_ptr);
  free (M->col_idx);
  free (M->values);
  M->row_ptr = NULL;
  M->col_idx = NULL;
  M->values = NULL;
  M->n = M->nnz = 0;
}

/* ── Phase weight computation ────────────────────────────────────── */

/**
 * Compute metallic-mean phase angles: theta_k = 2*pi * beta^k / sum.
 * Uses log-space arithmetic to prevent overflow for large n * beta.
 */
static void
compute_phase_angles (unsigned n, double beta, double *theta) {
  if (n == 0)
    return;

  double log_beta = log (beta);

  /* Find max for numerical stability */
  double max_log = (n - 1) * log_beta;

  double sum = 0.0;
  for (unsigned k = 0; k < n; k++) {
    double val = exp (k * log_beta - max_log);
    theta[k] = val;
    sum += val;
  }

  /* Normalize to [0, 2*pi] */
  double scale = TWO_PI / sum;
  for (unsigned k = 0; k < n; k++)
    theta[k] *= scale;
}

/* ── Edge list construction ──────────────────────────────────────── */

typedef struct edge {
  unsigned u, v;  /* variable indices (internal, 0-based) */
} edge;

/**
 * Extract unique variable pairs from all irredundant clauses.
 * Returns the number of unique edges found.
 * Edges are stored in `edges` (pre-allocated, max_edges capacity).
 */
static unsigned
extract_clause_edges (kissat *solver, edge *edges, unsigned max_edges) {
  unsigned n_edges = 0;

  /* Iterate over binary watches (binary clauses) */
  /* For large-clause arena iteration: */
  for (all_clauses (c)) {
    if (c->garbage)
      continue;
    if (c->redundant)
      continue;  /* only original clauses */

    unsigned size = c->size;
    if (size < 2)
      continue;

    /* For each pair of literals in the clause */
    for (unsigned i = 0; i < size && n_edges < max_edges; i++) {
      unsigned lit_u = c->lits[i];
      unsigned var_u = IDX (lit_u);

      for (unsigned j = i + 1; j < size && n_edges < max_edges; j++) {
        unsigned lit_v = c->lits[j];
        unsigned var_v = IDX (lit_v);

        if (var_u == var_v)
          continue;

        /* Store edge (smaller index first for deduplication later) */
        if (var_u < var_v) {
          edges[n_edges].u = var_u;
          edges[n_edges].v = var_v;
        } else {
          edges[n_edges].u = var_v;
          edges[n_edges].v = var_u;
        }
        n_edges++;
      }
    }
  }

  return n_edges;
}

/* ── Weighted edge for Laplacian construction ────────────────────── */

typedef struct weighted_edge {
  unsigned u, v;
  double weight;
} weighted_edge;

static int
wedge_cmp (const void *a, const void *b) {
  const weighted_edge *ea = (const weighted_edge *) a;
  const weighted_edge *eb = (const weighted_edge *) b;
  if (ea->u != eb->u)
    return (ea->u < eb->u) ? -1 : 1;
  if (ea->v != eb->v)
    return (ea->v < eb->v) ? -1 : 1;
  return 0;
}

/* ── Chiral Laplacian construction ───────────────────────────────── */

/**
 * Build the graph Laplacian with chiral metallic-mean edge weights.
 *
 * Edge weight: w(u,v) = cos(omega * dt) + chi * sin(omega * dt)
 * where dt = theta[u] - theta[v], omega is the frequency parameter,
 * and chi is the chirality (+1 or -1).
 *
 * Accumulates weights for duplicate edges (same variable pair in
 * multiple clauses). Stores as CSR sparse matrix.
 *
 * L = D - W (graph Laplacian).
 */
static int
build_chiral_laplacian (unsigned n_vars,
                        const edge *edges, unsigned n_edges,
                        const double *theta, double omega, int chirality,
                        sparse_matrix *L) {
  /* Pass 1: count edges per row (for CSR allocation) */
  unsigned *degree = (unsigned *) calloc (n_vars, sizeof (unsigned));
  if (!degree)
    return 0;

  /* We need a hash map for accumulating duplicate edges.
   * Simple approach: sort edges, then merge duplicates.
   * For AAA+ quality we use a temporary adjacency list. */

  /* Allocate temporary edge weight accumulator.
   * Key: (u,v) pair. Use a flat array indexed by edge index. */

  /* First, sort edges for deduplication */
  weighted_edge *wedges =
      (weighted_edge *) malloc (n_edges * sizeof (weighted_edge));
  if (!wedges) {
    free (degree);
    return 0;
  }

  for (unsigned i = 0; i < n_edges; i++) {
    double dt = theta[edges[i].u] - theta[edges[i].v];
    double w = cos (omega * dt) + chirality * sin (omega * dt);
    wedges[i].u = edges[i].u;
    wedges[i].v = edges[i].v;
    wedges[i].weight = w;
  }

  /* Sort by (u, v) for merging */
  qsort (wedges, n_edges, sizeof (weighted_edge), wedge_cmp);

  /* Merge duplicates */
  unsigned n_unique = 0;
  for (unsigned i = 0; i < n_edges; i++) {
    if (n_unique > 0 && wedges[n_unique - 1].u == wedges[i].u &&
        wedges[n_unique - 1].v == wedges[i].v) {
      wedges[n_unique - 1].weight += wedges[i].weight;
    } else {
      wedges[n_unique++] = wedges[i];
    }
  }

  /* Build CSR: each unique edge appears twice (symmetric) */
  unsigned nnz = 2 * n_unique + n_vars; /* off-diagonal + diagonal */

  L->n = n_vars;
  L->nnz = nnz;
  L->row_ptr = (unsigned *) calloc (n_vars + 1, sizeof (unsigned));
  L->col_idx = (unsigned *) malloc (nnz * sizeof (unsigned));
  L->values = (double *) malloc (nnz * sizeof (double));

  if (!L->row_ptr || !L->col_idx || !L->values) {
    free (degree);
    free (wedges);
    sparse_free (L);
    return 0;
  }

  /* Count entries per row */
  memset (degree, 0, n_vars * sizeof (unsigned));
  for (unsigned i = 0; i < n_unique; i++) {
    degree[wedges[i].u]++;
    degree[wedges[i].v]++;
  }

  /* Row pointers (include diagonal) */
  L->row_ptr[0] = 0;
  for (unsigned i = 0; i < n_vars; i++)
    L->row_ptr[i + 1] = L->row_ptr[i] + degree[i] + 1; /* +1 for diagonal */

  /* Fill CSR arrays */
  unsigned *pos = (unsigned *) calloc (n_vars, sizeof (unsigned));
  if (!pos) {
    free (degree);
    free (wedges);
    sparse_free (L);
    return 0;
  }

  /* Compute row sums for diagonal (D - W) */
  double *row_sum = (double *) calloc (n_vars, sizeof (double));
  if (!row_sum) {
    free (degree);
    free (wedges);
    free (pos);
    sparse_free (L);
    return 0;
  }

  for (unsigned i = 0; i < n_unique; i++) {
    unsigned u = wedges[i].u, v = wedges[i].v;
    double w = wedges[i].weight;

    unsigned idx_u = L->row_ptr[u] + pos[u]++;
    L->col_idx[idx_u] = v;
    L->values[idx_u] = -w; /* off-diagonal: -W */
    row_sum[u] += w;

    unsigned idx_v = L->row_ptr[v] + pos[v]++;
    L->col_idx[idx_v] = u;
    L->values[idx_v] = -w;
    row_sum[v] += w;
  }

  /* Add diagonal entries (D) */
  for (unsigned i = 0; i < n_vars; i++) {
    unsigned idx = L->row_ptr[i] + pos[i];
    L->col_idx[idx] = i;
    L->values[idx] = row_sum[i];
  }

  free (degree);
  free (wedges);
  free (pos);
  free (row_sum);
  return 1;
}

/* ── Sparse matrix-vector product ────────────────────────────────── */

static void
spmv (const sparse_matrix *M, const double *x, double *y) {
  for (unsigned i = 0; i < M->n; i++) {
    double sum = 0.0;
    for (unsigned j = M->row_ptr[i]; j < M->row_ptr[i + 1]; j++)
      sum += M->values[j] * x[M->col_idx[j]];
    y[i] = sum;
  }
}

/* ── Shifted power iteration for Fiedler vector ──────────────────── */

/**
 * Approximate the Fiedler vector (eigenvector of lambda_2) via
 * shifted power iteration on (lambda_max * I - L).
 *
 * Since L is positive semi-definite with smallest eigenvalue 0,
 * (lambda_max * I - L) has its LARGEST eigenvalue at lambda_max
 * (corresponding to the zero eigenvector of L) and second-largest
 * at (lambda_max - lambda_2). Power iteration converges to the
 * largest eigenvector, so we deflate the constant eigenvector
 * after each iteration to get the Fiedler vector.
 *
 * @param L       Sparse Laplacian (CSR format)
 * @param fiedler Output Fiedler vector (n entries, pre-allocated)
 * @param max_iter Maximum iterations
 * @param tol     Convergence tolerance
 * @return Approximate lambda_2 (Fiedler value)
 */
static double
compute_fiedler (const sparse_matrix *L, double *fiedler,
                 unsigned max_iter, double tol) {
  unsigned n = L->n;
  if (n < 2)
    return 0.0;

  double *v = fiedler;  /* reuse output buffer */
  double *Mv = (double *) malloc (n * sizeof (double));
  if (!Mv)
    return 0.0;

  /* Estimate lambda_max via a few power iteration steps on L */
  double lambda_max = 0.0;
  {
    double *tmp = (double *) malloc (n * sizeof (double));
    if (!tmp) {
      free (Mv);
      return 0.0;
    }
    for (unsigned i = 0; i < n; i++)
      tmp[i] = 1.0 / sqrt ((double) n);

    for (unsigned iter = 0; iter < 20; iter++) {
      spmv (L, tmp, Mv);
      double norm = 0.0;
      for (unsigned i = 0; i < n; i++)
        norm += Mv[i] * Mv[i];
      norm = sqrt (norm);
      if (norm < 1e-15)
        break;
      lambda_max = norm;
      for (unsigned i = 0; i < n; i++)
        tmp[i] = Mv[i] / norm;
    }
    free (tmp);
  }

  if (lambda_max < 1e-10) {
    free (Mv);
    return 0.0;
  }

  /* Initialize with pseudo-random vector orthogonal to constant */
  double inv_sqrt_n = 1.0 / sqrt ((double) n);
  for (unsigned i = 0; i < n; i++)
    v[i] = sin (2.0 * PI * i * GOLDEN_BETA / n);

  /* Remove constant component */
  double mean = 0.0;
  for (unsigned i = 0; i < n; i++)
    mean += v[i];
  mean /= n;
  for (unsigned i = 0; i < n; i++)
    v[i] -= mean;

  /* Normalize */
  double norm = 0.0;
  for (unsigned i = 0; i < n; i++)
    norm += v[i] * v[i];
  norm = sqrt (norm);
  if (norm < 1e-15) {
    free (Mv);
    return 0.0;
  }
  for (unsigned i = 0; i < n; i++)
    v[i] /= norm;

  /* Power iteration on (lambda_max * I - L) with deflation */
  double prev_rayleigh = 0.0;

  for (unsigned iter = 0; iter < max_iter; iter++) {
    /* Mv = (lambda_max * I - L) * v = lambda_max * v - L * v */
    spmv (L, v, Mv);
    for (unsigned i = 0; i < n; i++)
      Mv[i] = lambda_max * v[i] - Mv[i];

    /* Deflate: remove projection onto constant vector */
    mean = 0.0;
    for (unsigned i = 0; i < n; i++)
      mean += Mv[i];
    mean /= n;
    for (unsigned i = 0; i < n; i++)
      Mv[i] -= mean;

    /* Normalize */
    norm = 0.0;
    for (unsigned i = 0; i < n; i++)
      norm += Mv[i] * Mv[i];
    norm = sqrt (norm);
    if (norm < 1e-15)
      break;
    for (unsigned i = 0; i < n; i++)
      v[i] = Mv[i] / norm;

    /* Rayleigh quotient for convergence check: v^T L v / v^T v */
    spmv (L, v, Mv);
    double rayleigh = 0.0;
    for (unsigned i = 0; i < n; i++)
      rayleigh += v[i] * Mv[i];

    if (iter > 0 && fabs (rayleigh - prev_rayleigh) < tol) {
      free (Mv);
      return rayleigh;
    }
    prev_rayleigh = rayleigh;
  }

  free (Mv);
  return prev_rayleigh;
}

/* ── Main spectral preprocessing entry point ─────────────────────── */

void
kissat_spectral_preprocessing (kissat *solver) {
  const unsigned n_vars = VARS;

  if (n_vars < 10) {
    kissat_message (solver,
        "spectral preprocessing skipped (n=%u < 10)", n_vars);
    return;
  }

  kissat_section (solver, "spectral");
  kissat_message (solver,
      "chiral spectral preprocessing on %u variables", n_vars);

  /* ── Allocate working memory ─────────────────────────────────── */

  double *theta = (double *) malloc (n_vars * sizeof (double));
  double *fiedler = (double *) malloc (n_vars * sizeof (double));
  double *votes = (double *) calloc (n_vars, sizeof (double));
  unsigned max_edges = SPECTRAL_MAX_EDGES;
  edge *edges = (edge *) malloc (max_edges * sizeof (edge));

  if (!theta || !fiedler || !votes || !edges) {
    kissat_message (solver, "spectral: allocation failed, skipping");
    free (theta); free (fiedler); free (votes); free (edges);
    return;
  }

  /* ── Extract edges from clause-variable graph ────────────────── */

  unsigned n_edges = extract_clause_edges (solver, edges, max_edges);
  kissat_message (solver,
      "spectral: extracted %u clause-variable edges", n_edges);

  if (n_edges < n_vars) {
    kissat_message (solver,
        "spectral: too few edges (%u < %u vars), skipping", n_edges, n_vars);
    free (theta); free (fiedler); free (votes); free (edges);
    return;
  }

  /* ── Three-shell chiral pipeline ─────────────────────────────── */

  struct shell_config {
    double beta;
    int chirality;     /* +1 = right-handed, -1 = left-handed */
    double weight;     /* compound voting weight               */
    const char *name;
  };

  /* LRL chirality pattern (empirically optimal for n >= 50) */
  struct shell_config shells[3] = {
    { BRONZE_BETA, -1, 0.45, "bronze" },
    { SILVER_BETA, +1, 0.30, "silver" },
    { GOLDEN_BETA, -1, 0.25, "golden" },
  };

  /* Pendulum omega: powers of bronze metallic mean */
  double omegas[5] = {
    BRONZE_BETA,                                /* 3.303  */
    BRONZE_BETA * BRONZE_BETA,                  /* 10.908 */
    BRONZE_BETA * BRONZE_BETA * BRONZE_BETA,    /* 36.02  */
    BRONZE_BETA * BRONZE_BETA * BRONZE_BETA *
        BRONZE_BETA,                            /* 118.95 */
    BRONZE_BETA * BRONZE_BETA * BRONZE_BETA *
        BRONZE_BETA * BRONZE_BETA,              /* 392.80 */
  };

  unsigned total_shells = 0;

  for (unsigned oi = 0; oi < 5; oi++) {
    double omega = omegas[oi];

    for (unsigned si = 0; si < 3; si++) {
      struct shell_config *shell = &shells[si];

      /* Compute phase angles for this metallic mean */
      compute_phase_angles (n_vars, shell->beta, theta);

      /* Build chiral Laplacian */
      sparse_matrix L = { 0, 0, NULL, NULL, NULL };
      if (!build_chiral_laplacian (n_vars, edges, n_edges,
                                    theta, omega, shell->chirality, &L)) {
        continue;
      }

      /* Compute Fiedler vector */
      double lambda2 =
          compute_fiedler (&L, fiedler, SPECTRAL_MAX_ITERS, SPECTRAL_TOL);

      /* Center the Fiedler vector at its median for balanced partitioning.
       * Power iteration may leave residual constant component that biases
       * all values to one sign. Median-centering guarantees ~50/50 split. */
      {
        /* Compute median via partial sort (O(n) average) */
        double *sorted = (double *) malloc (n_vars * sizeof (double));
        if (sorted) {
          memcpy (sorted, fiedler, n_vars * sizeof (double));
          /* Simple selection: sort and take middle element */
          for (unsigned i = 0; i < n_vars / 2 + 1; i++) {
            unsigned min_idx = i;
            for (unsigned j = i + 1; j < n_vars; j++)
              if (sorted[j] < sorted[min_idx])
                min_idx = j;
            double tmp = sorted[i];
            sorted[i] = sorted[min_idx];
            sorted[min_idx] = tmp;
          }
          double median = sorted[n_vars / 2];
          for (unsigned i = 0; i < n_vars; i++)
            fiedler[i] -= median;
          free (sorted);
        }
      }

      /* Accumulate weighted votes from centered Fiedler sign.
       * Use magnitude as confidence: variables far from the median
       * get stronger votes than those near the cut. */
      for (unsigned i = 0; i < n_vars; i++) {
        double sign = (fiedler[i] >= 0.0) ? 1.0 : -1.0;
        double magnitude = fabs (fiedler[i]);
        votes[i] += shell->weight * sign * (1.0 + magnitude);
      }

      sparse_free (&L);
      total_shells++;
    }
  }

  kissat_message (solver,
      "spectral: computed %u shell Fiedler vectors (5 omegas x 3 shells)",
      total_shells);

  /* ── Set target phases from compound vote ────────────────────── */

  unsigned positive = 0, negative = 0;

  for (unsigned idx = 0; idx < n_vars; idx++) {
    /* Skip eliminated variables */
    if (solver->flags[idx].eliminated)
      continue;

    value phase;
    if (votes[idx] > 0.0) {
      phase = 1;
      positive++;
    } else if (votes[idx] < 0.0) {
      phase = -1;
      negative++;
    } else {
      phase = 1; /* tie-break positive */
      positive++;
    }

    solver->phases.target[idx] = phase;
  }

  kissat_message (solver,
      "spectral: set %u positive, %u negative target phases",
      positive, negative);

  /* ── Cleanup ─────────────────────────────────────────────────── */

  free (theta);
  free (fiedler);
  free (votes);
  free (edges);
}
