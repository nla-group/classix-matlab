/* Bare bones (full vector).'*sparse submatrix                          */
/* Produces a full vector result                                        */
/* Assumes all input elements are finite                                */
/* E.g., does not account for NaN or 0*inf etc.                         */
/* Based on code by James Tursa, modified by Stefan Guettel             */

#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ptrdiff_t nn, j, k, nrow;
    double *Mpr, *Vpr, *Cpr;   /* pointers to input & output matrices*/
    int col_s, col_e;    /* start & end indices of matrix columns to use */
    mwIndex *Mir, *Mjc;

    /*
    m = mxGetM(prhs[1]);  
    n = mxGetN(prhs[1]);
    */
    

    col_s = (int) mxGetScalar(prhs[2]);
    col_e = (int) mxGetScalar(prhs[3]);
    nn = (ptrdiff_t) (col_e - col_s + 1);

/* Create output */
    plhs[0] = mxCreateDoubleMatrix( 1, (ptrdiff_t) nn, mxREAL ); 

/* Get data pointers */
    Vpr = mxGetPr(prhs[0]);   /* vector */
    Mpr = mxGetPr(prhs[1]);   /* sparse matrix */
    Mir = mxGetIr(prhs[1]);
    Mjc = mxGetJc(prhs[1]);
    Cpr = mxGetPr(plhs[0]);   /* output vector */
/* Calculate result */
    for( j=0; j<col_e; j++ ) {
        nrow = Mjc[j+1] - Mjc[j];  /* Number of elements for this column */

        /* Cpr[j] = 0; */   /* mxCreateDoubleMatrix takes care of init */

        if( j+1<col_s ) {
            /* skip this column */


            /*
            while( nrow-- ) {
                *Mpr++;
                *Mir++;
            }
            */

            Mpr += nrow;   /* skip matrix element */
            Mir += nrow;  /* skip row-index position for this element */
            
            continue;
        }

        k = j + 1 - col_s; 
        while( nrow-- ) {
            Cpr[k] += *Mpr++ * Vpr[*Mir++];  /* Accumulate contribution of j-th column */
        }
    }
}