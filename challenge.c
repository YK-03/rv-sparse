/*
 * RV-Sparse: Sparse Matrix-Vector Multiplication
 * ================================================
 * Implements sparse_multiply():
 *   1. Scans row-major matrix A for non-zero elements.
 *   2. Extracts them into Compressed Sparse Row (CSR) format
 *      using caller-provided buffers (zero dynamic allocation).
 *   3. Computes y = A * x using the CSR data.
 *   4. Writes results into caller-provided output buffer y.
 *
 * Build:  gcc -lm -O2 -o run challenge.c
 * Run:    ./run
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

/* ─── Types ──────────────────────────────────────────────────────────────── */

/*
 * CSR representation:
 *
 *   values[nnz]      — non-zero element values, in row-major order
 *   col_idx[nnz]     — column index of each value
 *   row_ptr[rows+1]  — row_ptr[i] = index in values[] where row i starts;
 *                       row_ptr[rows] = nnz (sentinel)
 *
 * This is the standard CSR layout used by SuiteSparse, MKL, cuSPARSE, etc.
 */
typedef struct {
    double  *values;   /* caller-allocated, length >= nnz          */
    int     *col_idx;  /* caller-allocated, length >= nnz          */
    int     *row_ptr;  /* caller-allocated, length >= (rows + 1)   */
    int      rows;
    int      cols;
    int      nnz;      /* number of non-zeros (filled by compress) */
} CSRMatrix;

/* ─── Core function ──────────────────────────────────────────────────────── */

/*
 * sparse_multiply()
 *
 * Parameters
 * ----------
 * A        : row-major dense matrix, shape [rows x cols]
 * rows     : number of rows in A
 * cols     : number of columns in A
 * x        : input vector, length cols
 * y        : output vector, length rows  (caller-allocated, zeroed on entry)
 * csr      : CSRMatrix whose buffer pointers are already allocated by the
 *            caller; this function fills values/col_idx/row_ptr and nnz.
 *
 * Guarantees
 * ----------
 * - Zero dynamic memory allocation (no malloc/calloc/realloc inside).
 * - O(rows * cols) scan for compression, O(nnz) for multiply.
 * - Numerically identical to dense y = A*x for non-zero elements.
 */
void sparse_multiply(const double *A,
                     int           rows,
                     int           cols,
                     const double *x,
                     double       *y,
                     CSRMatrix    *csr)
{
    /* ── Phase 1: Compress A into CSR ───────────────────────────────────── */
    int nnz = 0;

    for (int i = 0; i < rows; i++) {
        csr->row_ptr[i] = nnz;                      /* start of row i       */

        for (int j = 0; j < cols; j++) {
            double val = A[i * cols + j];
            if (val != 0.0) {                        /* non-zero test        */
                csr->values[nnz]  = val;
                csr->col_idx[nnz] = j;
                nnz++;
            }
        }
    }

    csr->row_ptr[rows] = nnz;                        /* CSR sentinel         */
    csr->rows = rows;
    csr->cols = cols;
    csr->nnz  = nnz;

    /* ── Phase 2: Sparse matrix-vector product y = A * x ───────────────── */
    /*
     * For each row i, dot-product the stored non-zeros with x.
     * y is assumed to be pre-zeroed by the caller.
     */
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        int start = csr->row_ptr[i];
        int end   = csr->row_ptr[i + 1];

        for (int k = start; k < end; k++) {
            sum += csr->values[k] * x[csr->col_idx[k]];
        }

        y[i] = sum;
    }
}

/* ─── Test harness ───────────────────────────────────────────────────────── */

/* Compute dense y = A*x for reference comparison */
static void dense_multiply(const double *A, int rows, int cols,
                            const double *x, double *y)
{
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++)
            sum += A[i * cols + j] * x[j];
        y[i] = sum;
    }
}

/* Check two vectors agree within tolerance */
static int vectors_close(const double *a, const double *b, int n, double tol)
{
    for (int i = 0; i < n; i++)
        if (fabs(a[i] - b[i]) > tol) return 0;
    return 1;
}

/* Print a dense matrix */
static void print_matrix(const char *label, const double *M, int rows, int cols)
{
    printf("%s (%dx%d):\n", label, rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("  [");
        for (int j = 0; j < cols; j++)
            printf(" %6.2f", M[i * cols + j]);
        printf(" ]\n");
    }
}

/* Print a vector */
static void print_vector(const char *label, const double *v, int n)
{
    printf("%s: [", label);
    for (int i = 0; i < n; i++)
        printf(" %6.2f", v[i]);
    printf(" ]\n");
}

/* Print CSR internals */
static void print_csr(const CSRMatrix *csr)
{
    printf("CSR (nnz=%d):\n", csr->nnz);
    printf("  row_ptr: [");
    for (int i = 0; i <= csr->rows; i++) printf(" %d", csr->row_ptr[i]);
    printf(" ]\n  col_idx: [");
    for (int k = 0; k < csr->nnz; k++) printf(" %d", csr->col_idx[k]);
    printf(" ]\n  values:  [");
    for (int k = 0; k < csr->nnz; k++) printf(" %.2f", csr->values[k]);
    printf(" ]\n");
}

/* ── Test 1: Basic 3x3 sparse matrix ─────────────────────────────────────── */
static int test_basic(void)
{
    printf("=== Test 1: Basic 3x3 ===\n");

    const int ROWS = 3, COLS = 3;
    double A[] = {
        1.0, 0.0, 2.0,
        0.0, 3.0, 0.0,
        4.0, 0.0, 5.0
    };
    double x[]    = { 1.0, 2.0, 3.0 };
    double y[3]   = { 0 };
    double ref[3] = { 0 };

    /* Max nnz = ROWS*COLS in the worst case */
    double  vals[9];
    int     cols[9], rptr[4];
    CSRMatrix csr = { vals, cols, rptr, 0, 0, 0 };

    sparse_multiply(A, ROWS, COLS, x, y, &csr);
    dense_multiply(A, ROWS, COLS, x, ref);

    print_matrix("A", A, ROWS, COLS);
    print_vector("x", x, COLS);
    print_csr(&csr);
    print_vector("y (sparse)", y, ROWS);
    print_vector("y (dense) ", ref, ROWS);

    int ok = vectors_close(y, ref, ROWS, 1e-10);
    printf("Result: %s\n\n", ok ? "PASS" : "FAIL");
    return ok;
}

/* ── Test 2: All-zeros matrix ────────────────────────────────────────────── */
static int test_all_zeros(void)
{
    printf("=== Test 2: All-zeros 4x4 ===\n");

    const int ROWS = 4, COLS = 4;
    double A[16] = { 0 };
    double x[]   = { 1.0, 2.0, 3.0, 4.0 };
    double y[4]  = { 0 };
    double ref[4]= { 0 };

    double  vals[16];
    int     cols[16], rptr[5];
    CSRMatrix csr = { vals, cols, rptr, 0, 0, 0 };

    sparse_multiply(A, ROWS, COLS, x, y, &csr);
    dense_multiply(A, ROWS, COLS, x, ref);

    int ok = (csr.nnz == 0) && vectors_close(y, ref, ROWS, 1e-10);
    printf("nnz=%d  Result: %s\n\n", csr.nnz, ok ? "PASS" : "FAIL");
    return ok;
}

/* ── Test 3: Dense matrix (no zeros) ─────────────────────────────────────── */
static int test_dense(void)
{
    printf("=== Test 3: Fully dense 3x3 ===\n");

    const int ROWS = 3, COLS = 3;
    double A[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };
    double x[]   = { 1.0, 0.0, -1.0 };
    double y[3]  = { 0 };
    double ref[3]= { 0 };

    double  vals[9];
    int     cols[9], rptr[4];
    CSRMatrix csr = { vals, cols, rptr, 0, 0, 0 };

    sparse_multiply(A, ROWS, COLS, x, y, &csr);
    dense_multiply(A, ROWS, COLS, x, ref);

    int ok = (csr.nnz == 9) && vectors_close(y, ref, ROWS, 1e-10);
    printf("nnz=%d  Result: %s\n\n", csr.nnz, ok ? "PASS" : "FAIL");
    return ok;
}

/* ── Test 4: Single non-zero per row (diagonal) ──────────────────────────── */
static int test_diagonal(void)
{
    printf("=== Test 4: Diagonal 4x4 ===\n");

    const int ROWS = 4, COLS = 4;
    double A[] = {
        2.0, 0.0, 0.0, 0.0,
        0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 5.0, 0.0,
        0.0, 0.0, 0.0, 7.0
    };
    double x[]   = { 1.0, 2.0, 3.0, 4.0 };
    double y[4]  = { 0 };
    double ref[4]= { 0 };

    double  vals[16];
    int     cols[16], rptr[5];
    CSRMatrix csr = { vals, cols, rptr, 0, 0, 0 };

    sparse_multiply(A, ROWS, COLS, x, y, &csr);
    dense_multiply(A, ROWS, COLS, x, ref);

    int ok = (csr.nnz == 4) && vectors_close(y, ref, ROWS, 1e-10);
    printf("nnz=%d  Result: %s\n\n", csr.nnz, ok ? "PASS" : "FAIL");
    return ok;
}

/* ── Test 5: Non-square matrix (2x5) ─────────────────────────────────────── */
static int test_non_square(void)
{
    printf("=== Test 5: Non-square 2x5 ===\n");

    const int ROWS = 2, COLS = 5;
    double A[] = {
        0.0, 1.5, 0.0, 0.0, 2.5,
        3.0, 0.0, 0.0, 4.0, 0.0
    };
    double x[]   = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    double y[2]  = { 0 };
    double ref[2]= { 0 };

    double  vals[10];
    int     cols[10], rptr[3];
    CSRMatrix csr = { vals, cols, rptr, 0, 0, 0 };

    sparse_multiply(A, ROWS, COLS, x, y, &csr);
    dense_multiply(A, ROWS, COLS, x, ref);

    int ok = (csr.nnz == 4) && vectors_close(y, ref, ROWS, 1e-10);
    printf("nnz=%d  Result: %s\n\n", csr.nnz, ok ? "PASS" : "FAIL");
    return ok;
}

/* ── Test 6: Negative and floating-point values ──────────────────────────── */
static int test_negative_floats(void)
{
    printf("=== Test 6: Negative & float values ===\n");

    const int ROWS = 3, COLS = 3;
    double A[] = {
        -1.5,  0.0,  3.7,
         0.0, -2.2,  0.0,
         0.5,  0.0, -0.8
    };
    double x[]   = { 2.0, -1.0, 4.0 };
    double y[3]  = { 0 };
    double ref[3]= { 0 };

    double  vals[9];
    int     cols[9], rptr[4];
    CSRMatrix csr = { vals, cols, rptr, 0, 0, 0 };

    sparse_multiply(A, ROWS, COLS, x, y, &csr);
    dense_multiply(A, ROWS, COLS, x, ref);

    int ok = vectors_close(y, ref, ROWS, 1e-10);
    printf("nnz=%d  Result: %s\n\n", csr.nnz, ok ? "PASS" : "FAIL");
    return ok;
}

/* ── Test 7: Large sparse matrix (stress test) ───────────────────────────── */
static int test_large(void)
{
    printf("=== Test 7: Large 100x100 (~5%% density) ===\n");

    const int ROWS = 100, COLS = 100;
    /* Stack-allocate everything — no malloc */
    static double A[100 * 100];
    static double x[100], y[100], ref[100];
    static double vals[100 * 100];
    static int    colidx[100 * 100], rptr[101];

    memset(A,   0, sizeof(A));
    memset(y,   0, sizeof(y));
    memset(ref, 0, sizeof(ref));

    /* Seed with a simple pattern: non-zero at positions where (i+j) % 20 == 0 */
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++)
            if ((i + j) % 20 == 0)
                A[i * COLS + j] = (double)((i * 3 + j * 7) % 13 + 1);

    for (int j = 0; j < COLS; j++)
        x[j] = (double)(j % 5 + 1);

    CSRMatrix csr = { vals, colidx, rptr, 0, 0, 0 };

    sparse_multiply(A, ROWS, COLS, x, y, &csr);
    dense_multiply(A, ROWS, COLS, x, ref);

    int ok = vectors_close(y, ref, ROWS, 1e-8);
    printf("nnz=%d  Result: %s\n\n", csr.nnz, ok ? "PASS" : "FAIL");
    return ok;
}

/* ─── main ───────────────────────────────────────────────────────────────── */
int main(void)
{
    printf("RV-Sparse: Sparse Matrix-Vector Multiplication\n");
    printf("===============================================\n\n");

    int passed = 0, total = 7;

    passed += test_basic();
    passed += test_all_zeros();
    passed += test_dense();
    passed += test_diagonal();
    passed += test_non_square();
    passed += test_negative_floats();
    passed += test_large();

    printf("===============================================\n");
    printf("Results: %d / %d tests passed\n", passed, total);

    return (passed == total) ? EXIT_SUCCESS : EXIT_FAILURE;
}
