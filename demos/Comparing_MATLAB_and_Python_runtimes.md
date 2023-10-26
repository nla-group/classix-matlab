
# <span style="color:rgb(213,80,0)">Comparing MATLAB and Python runtimes</span>

One of the experiments in the CLASSIX paper [1] compares several clustering algorithms on the URI machine learning repository [2]. The Python code that generated these results, together with all the used hyperparameters, is available in the CLASSIX GitHub repository: [https://github.com/nla-group/classix/blob/master/exp/run_real_world.py](https://github.com/nla-group/classix/blob/master/exp/run_real_world.py) 


Here we use these datasets to compare the runtimes of the MATLAB and Python implementations of CLASSIX. For the MATLAB, we test two variants: (i) pure MATLAB implementation, and (ii) using MATLAB with the MEX file <samp>matxsubmat.c</samp> to speed up submatrix-times-vector products. 


Note that the below requires a working MATLAB-Python link.

```matlab
clear all
addpath ..
%ari = @(a,b) py.sklearn.metrics.adjusted_rand_score(a,b);
ari = @(a,b) rand_index(double(a),double(b),'adjusted');
```
# Banknote dataset
```matlab
ret = py.classix.loadData('Banknote');
data = double(ret{1});
labels = double(ret{2});
% z-normalization used in https://github.com/nla-group/classix/blob/master/exp/run_real_world.py
data = (data - mean(data))./std(data); 

% MATLAB CLASSIX
no_mex = struct('use_mex',0);
tic
[label, explain, out] = classix(data,0.21,41,no_mex);
fprintf('CLASSIX.M       runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(label)),ari(labels,label))
```

```TextOutput
CLASSIX.M       runtime: 0.020 seconds - classes: 2 - ARI: 0.87
```

```matlab
% MATLAB CLASSIX (MEX)
tic
[label, explain, out] = classix(data,0.21,41);
fprintf('CLASSIX.M (MEX) runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(label)),ari(labels,label))
```

```TextOutput
CLASSIX.M (MEX) runtime: 0.021 seconds - classes: 2 - ARI: 0.87
```

```matlab
% Python CLASSIX
tic
clx = py.classix.CLASSIX(radius=0.21, verbose=0, minPts=int32(41));
clx = clx.fit(data);
fprintf('CLASSIX.PY      runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(double(clx.labels_))),ari(labels,clx.labels_))
```

```TextOutput
CLASSIX.PY      runtime: 0.050 seconds - classes: 2 - ARI: 0.87
```
# Dermatology dataset
```matlab
ret = py.classix.loadData('Dermatology');
data = double(ret{1});
data = data(:,1:33); % final column has NaN's?
labels = double(ret{2});
data = (data - mean(data))./std(data); 

% MATLAB CLASSIX 
tic
[label, explain, out] = classix(data,0.4,4,no_mex);
fprintf('CLASSIX.M       runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(label)),ari(labels,label))
```

```TextOutput
CLASSIX.M       runtime: 0.025 seconds - classes: 6 - ARI: 0.45
```

```matlab
% MATLAB CLASSIX (MEX)
tic
[label, explain, out] = classix(data,0.4,4);
fprintf('CLASSIX.M (MEX) runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(label)),ari(labels,label))
```

```TextOutput
CLASSIX.M (MEX) runtime: 0.024 seconds - classes: 6 - ARI: 0.45
```

```matlab
% Python CLASSIX
tic
clx = py.classix.CLASSIX(radius=0.4, verbose=0, minPts=int32(4));
clx = clx.fit(data);
fprintf('CLASSIX.PY      runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(double(clx.labels_))),ari(labels,clx.labels_))
```

```TextOutput
CLASSIX.PY      runtime: 0.072 seconds - classes: 6 - ARI: 0.45
```
# Ecoli dataset
```matlab
ret = py.classix.loadData('Ecoli');
data = double(ret{1});
labels = double(ret{2});
data = (data - mean(data))./std(data); 

% MATLAB CLASSIX
tic
[label, explain, out] = classix(data,0.3,4,no_mex);
fprintf('CLASSIX.M       runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(label)),ari(labels,label))
```

```TextOutput
CLASSIX.M       runtime: 0.014 seconds - classes: 7 - ARI: 0.56
```

```matlab
% MATLAB CLASSIX (MEX) 
tic
[label, explain, out] = classix(data,0.3,4);
fprintf('CLASSIX.M (MEX) runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(label)),ari(labels,label))
```

```TextOutput
CLASSIX.M (MEX) runtime: 0.014 seconds - classes: 7 - ARI: 0.56
```

```matlab
% Python CLASSIX
tic
clx = py.classix.CLASSIX(radius=0.3, verbose=0, minPts=int32(4));
clx = clx.fit(data);
fprintf('CLASSIX.PY      runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(double(clx.labels_))),ari(labels,clx.labels_))
```

```TextOutput
CLASSIX.PY      runtime: 0.030 seconds - classes: 7 - ARI: 0.56
```
# Glass dataset
```matlab
ret = py.classix.loadData('Glass');
data = double(ret{1});
labels = double(ret{2});
data = (data - mean(data))./std(data); 

% MATLAB CLASSIX
tic
[label, explain, out] = classix(data,0.725,1,no_mex);
fprintf('CLASSIX.M       runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(label)),ari(labels,label))
```

```TextOutput
CLASSIX.M       runtime: 0.010 seconds - classes: 26 - ARI: 0.23
```

```matlab
% MATLAB CLASSIX (MEX)
tic
[label, explain, out] = classix(data,0.725,1);
fprintf('CLASSIX.M (MEX) runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(label)),ari(labels,label))
```

```TextOutput
CLASSIX.M (MEX) runtime: 0.009 seconds - classes: 26 - ARI: 0.23
```

```matlab
% Python CLASSIX
tic
clx = py.classix.CLASSIX(radius=0.725, verbose=0, minPts=int32(1));
clx = clx.fit(data);
fprintf('CLASSIX.PY      runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(double(clx.labels_))),ari(labels,clx.labels_))
```

```TextOutput
CLASSIX.PY      runtime: 0.017 seconds - classes: 26 - ARI: 0.23
```
# Iris dataset
```matlab
ret = py.classix.loadData('Iris');
data = double(ret{1});
labels = double(ret{2});
data = (data - mean(data))./std(data); 

% MATLAB CLASSIX
tic
[label, explain, out] = classix(data,0.225,4,no_mex);
fprintf('CLASSIX.M       runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(label)),ari(labels,label))
```

```TextOutput
CLASSIX.M       runtime: 0.007 seconds - classes: 4 - ARI: 0.56
```

```matlab
% MATLAB CLASSIX (MEX) 
tic
[label, explain, out] = classix(data,0.225,4);
fprintf('CLASSIX.M (MEX) runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(label)),ari(labels,label))
```

```TextOutput
CLASSIX.M (MEX) runtime: 0.006 seconds - classes: 4 - ARI: 0.56
```

```matlab
% Python CLASSIX
tic
clx = py.classix.CLASSIX(radius=0.225, verbose=0, minPts=int32(4));
clx = clx.fit(data);
fprintf('CLASSIX.PY      runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(double(clx.labels_))),ari(labels,clx.labels_))
```

```TextOutput
CLASSIX.PY      runtime: 0.013 seconds - classes: 4 - ARI: 0.56
```
# Seeds dataset
```matlab
ret = py.classix.loadData('Seeds');
data = double(ret{1});
labels = double(ret{2});
data = (data - mean(data))./std(data); 

% MATLAB CLASSIX
tic
[label, explain, out] = classix(data,0.15,9,no_mex);
fprintf('CLASSIX.M       runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(label)),ari(labels,label))
```

```TextOutput
CLASSIX.M       runtime: 0.010 seconds - classes: 3 - ARI: 0.70
```

```matlab
% MATLAB CLASSIX (MEX) 
tic
[label, explain, out] = classix(data,0.15,9);
fprintf('CLASSIX.M (MEX) runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(label)),ari(labels,label))
```

```TextOutput
CLASSIX.M (MEX) runtime: 0.011 seconds - classes: 3 - ARI: 0.70
```

```matlab
% Python CLASSIX
tic
clx = py.classix.CLASSIX(radius=0.15, verbose=0, minPts=int32(9));
clx = clx.fit(data);
fprintf('CLASSIX.PY      runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(double(clx.labels_))),ari(labels,clx.labels_))
```

```TextOutput
CLASSIX.PY      runtime: 0.027 seconds - classes: 3 - ARI: 0.70
```
# Wine dataset
```matlab
ret = py.classix.loadData('Wine');
data = double(ret{1});
labels = double(ret{2});
data = (data - mean(data))./std(data); 

% MATLAB CLASSIX
tic
[label, explain, out] = classix(data,0.425,4,no_mex);
fprintf('CLASSIX.M       runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(label)),ari(labels,label))
```

```TextOutput
CLASSIX.M       runtime: 0.010 seconds - classes: 2 - ARI: 0.47
```

```matlab
% MATLAB CLASSIX (MEX)
tic
[label, explain, out] = classix(data,0.425,4);
fprintf('CLASSIX.M (MEX) runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(label)),ari(labels,label))
```

```TextOutput
CLASSIX.M (MEX) runtime: 0.011 seconds - classes: 2 - ARI: 0.47
```

```matlab
% Python CLASSIX
tic
clx = py.classix.CLASSIX(radius=0.425, verbose=0, minPts=int32(4));
clx = clx.fit(data);
fprintf('CLASSIX.PY      runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(double(clx.labels_))),ari(labels,clx.labels_))
```

```TextOutput
CLASSIX.PY      runtime: 0.035 seconds - classes: 2 - ARI: 0.47
```
# Phoneme dataset
```matlab
ret = py.classix.loadData('Phoneme');
data = double(ret{1});
labels = double(ret{2});
data = (data - mean(data))./std(data); 

% MATLAB CLASSIX
tic
[label, explain, out] = classix(data,0.445,8,no_mex);
fprintf('CLASSIX.M       runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(label)),ari(labels,label))
```

```TextOutput
CLASSIX.M       runtime: 4.965 seconds - classes: 4 - ARI: 0.76
```

```matlab
% MATLAB CLASSIX (MEX)
tic
[label, explain, out] = classix(data,0.445,8);
fprintf('CLASSIX.M (MEX) runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(label)),ari(labels,label))
```

```TextOutput
CLASSIX.M (MEX) runtime: 2.262 seconds - classes: 4 - ARI: 0.76
```

```matlab
% Python CLASSIX
tic
clx = py.classix.CLASSIX(radius=0.445, verbose=0, minPts=int32(8));
clx = clx.fit(data);
fprintf('CLASSIX.PY      runtime: %5.3f seconds - classes: %d - ARI: %3.2f\n',...
    toc,length(unique(double(clx.labels_))),ari(labels,clx.labels_))
```

```TextOutput
CLASSIX.PY      runtime: 3.472 seconds - classes: 4 - ARI: 0.76
```
# VDU Signals dataset
```matlab
ret = py.classix.loadData('vdu_signals');
data = double(ret);
data = (data - mean(data))./std(data); 

% MATLAB CLASSIX
tic
[label, explain, out] = classix(data,0.3,1,no_mex);
fprintf('CLASSIX.M       runtime: %5.3f seconds - classes: %d\n',...
    toc,length(unique(label)))
```

```TextOutput
CLASSIX.M       runtime: 1.331 seconds - classes: 11
```

```matlab
% MATLAB CLASSIX (MEX)
tic
[label, explain, out] = classix(data,0.3,1);
fprintf('CLASSIX.M (MEX) runtime: %5.3f seconds - classes: %d\n',...
    toc,length(unique(label)))
```

```TextOutput
CLASSIX.M (MEX) runtime: 1.158 seconds - classes: 11
```

```matlab
% Python CLASSIX
tic
clx = py.classix.CLASSIX(radius=0.3, verbose=0, minPts=int32(1));
clx = clx.fit(data);
fprintf('CLASSIX.PY      runtime: %5.3f seconds - classes: %d\n',...
    toc,length(unique(double(clx.labels_))))
```

```TextOutput
CLASSIX.PY      runtime: 3.513 seconds - classes: 9
```

```matlab
fprintf('Agreement between M and PY: ARI %3.2f\n',ari(label,clx.labels_))
```

```TextOutput
Agreement between M and PY: ARI 1.00
```
# Learn more about CLASSIX?

CLASSIX is a fast and memory-efficient clustering algorithm which produces explainable results. If you'd like to learn more, here are a couple of online resources:

-  arXiv paper: [Fast and explainable clustering based on sorting (arxiv.org)](https://arxiv.org/abs/2202.01456) 
-  Python code: [Fast and explainable clustering based on sorting (github.com)](https://github.com/nla-group/classix) 
-  YouTube video: [CLASSIX - Fast and explainable clustering based on sorting - YouTube](https://www.youtube.com/watch?v=K94zgRjFEYo) 
# References

[1] C. Chen and S. Güttel. "Fast and explainable clustering based on sorting." arXiv: [https://arxiv.org/abs/2202.01456](https://arxiv.org/abs/2202.01456), 2022.


[2] D. Dua and C. Graff. "UCI machine learning repository." URL: [http://archive.ics.uci.edu/ml](http://archive.ics.uci.edu/ml), 2017.

