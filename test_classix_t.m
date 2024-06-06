
clear all
close all hidden
load data/BL_Sets.mat
data = sparse(double(data));

%%
radius = 0.39; 
minPts = 50;
opts.merge_scale = 1.2;
opts.merge_tiny_groups = 1;
opts.use_mex = 1;

tic
[clab, explain, out] = classix_t(data, radius, minPts, opts);
toc
ri = rand_index(clab, labels, 'adjusted')