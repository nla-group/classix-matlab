% vecxspsubmat computes the product of a dense vector b and 
% a submatrix of the sparse matrix A  without creating a memory copy.  
%
% c = vecxspsubmat(b,A,ind1,ind2)  

rng('default')
try 
    vecxspsubmat(1,sparse(1),1,1);
catch
    try
        mex vecxspsubmat.c 
    catch
        error('vecxspsubmat.c could not be compiled')
    end
end

%%
A = sparse(randn(100,500));
b = randn(100,1);
tic; c1 = b'*A(:,1:100); toc
tic; c2 = vecxspsubmat(b,A,1,100); toc
norm(c1 - c2)

%%
load data/BL_Sets.mat
A = sparse(double(data)).';
b = randn(1024,1);
tic; c0 = b'*A; toc
tic; c1 = b'*A(:,1:50000); toc
tic; c2 = vecxspsubmat(b,A,1,50000); toc
