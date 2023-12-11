% matxsubmat computes the matrix product of a matrix A   
% and a submatrix of B without creating a memory copy.  
% It calls the BLAS function dgemm.
%
% C = matxsubmat(A,B,ind1,ind2)  
% computes the product C = A*B(:,ind1:ind2)
% where A, B, and C are matrices containing real values.

rng('default')
try 
    matxsubmat(1,1,1,1);
catch
    try
        mex matxsubmat.c -lmwblas
    catch
        error('matxsubmat.c could not be compiled')
    end
end

A = randn(1,100);
B = randn(100,10000);
tic; C1 = A*B(:,5000:10000); toc
tic; C2 = matxsubmat(A,B,5000,10000); toc
norm(C1 - C2)
