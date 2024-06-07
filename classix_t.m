function [label, explain, out] = classix_t(data, radius, minPts, opts)
% CLASSIX_T - Fast and explainable clustering in the Tanimoto distance.
% [label, explain, out] = classix_t(data, radius, minPts, opts)
%
% inputs   * data matrix (each row is a data point)
%          * radius parameter between 0 and 1
%          * minPts parameter (default 1)
%          * opts structure (optional) with fields
%                 .merge_tiny_groups - Boolean default 1
%                 .use_mex - Boolean default 1
%                 .merge_scale - merge scaling param default 1/sqrt(radius)
%
% returns  * cluster labels of the data
%          * function handle to explain functionality
%          * out structure with fields
%                .cs    -  cluster size (#points in each cluster)
%                .dist  -  #distance computations during aggregation
%                .gc    -  group center indices 
%                .t1... -  timings of CLASSIX's phases (in seconds)
%
% This is a MATLAB implementation of the CLASSIX clustering algorithm:
%   X. Chen & S. Güttel. Fast and explainable clustering based on sorting, 
%   Pattern Recognition, 150: 110298, 2024.
%   https://arxiv.org/abs/2202.01456
% adapted to use the Tanimoto distance.
%
% The code optionally uses MEX implementations of a more efficent
% submatrix multiplication (avoiding memory copying). To enable that, 
% you need to compile 
% 
%   mex matxsubmat.c -lmwblas
%   mex vecxspsubmat.c


% preparation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t = tic;

if nargin < 4
    opts = struct();
end
if isfield(opts,'merge_tiny_groups')
    merge_tiny_groups = opts.merge_tiny_groups;
else
    merge_tiny_groups = 1; % merge groups with < minPts points
end
if isfield(opts,'use_mex')
    use_mex = opts.use_mex;
else
    use_mex = 1;
end
if isfield(opts,'merge_scale')
    merge_scale = opts.merge_scale;
else
    merge_scale = 1/sqrt(radius);
end
if radius <= 0 || radius > 1
    error('radius needs to be between 0 and 1');
end
if merge_scale*radius > 1
    error('merge_scale*radius needs to be <= 1');
end
if nargin < 3
    minPts = 1;
end
if size(data,1) < size(data,2)
    warning('Fewer data points than features. Check that each row corresponds to a data point.');
end
if size(data,2) > 5000
    warning('More than 5000 features. Consider applying some dimension reduction first.');
end
if use_mex

    if issparse(data)
        try
            vecxspsubmat(1,sparse(1),1,1);
            use_mex = 1;  % yes, use vecxspsubmat MEX file
        catch
            use_mex = 0; 
            disp('MEX file not found. Consider compiling vecxspsubmat.c via')
            disp('mex vecxspsubmat.c')
            disp('or remove this warning with opts.use_mex=0.')
        end
    else
        try
            matxsubmat(1,1,1,1);
            use_mex = 1;  % yes, use matxsubmat MEX file
        catch
            use_mex = 0; 
            disp('MEX file not found. Consider compiling matxsubmat.c via')
            disp('mex matxsubmat.c -lmwblas')
            disp('or remove this warning with opts.use_mex=0.')
        end
    end
end

x = double(data');  % transpose. much faster when data points are stored column-wise
U = full([ sum(x,1).', sum(x(1:2:end,:),1).' ]);
u = U(:,1);                  % scores
[u,ind] = sort(u); u = u.';
x = x(:,ind);

out = struct();
out.t1_prepare = toc(t);

%% aggregation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t = tic;

n = size(x,2);
label = zeros(n,1);
lab = 1;
dist = 0;    % # distance comput.
i = 0;
gc = [];     % indices of group centers (in sorted array)
gs = [];     % group size
wb = waitbar(0,'CLASSIX\_T Aggregation');
while i < n
    i = i + 1;
    
    if label(i) > 0
        continue
    end
    label(i) = lab;
    gc = [gc, i];
    gs = [gs, 1];
    xi = full(x(:,i));
    ui = u(i);

    % exclusion u > u(i)/(1-radius)
    
    last_j = n;
    if use_mex
        % precomp inner products ips = xi'*x(:,i+1:last_j)
        % TODO: could exploit that u is sorted (binary search)
        last_j = find(u <= ui/(1-radius), 1, 'last'); 

        if issparse(x)
            ips = vecxspsubmat(xi,x,i+1,last_j);
        else
            ips = matxsubmat(xi',x,i+1,last_j);
        end
        dist = dist + last_j - i;

        % Note: The number of distance calculation can be
        % slightly larger than without using mex because
        % we're not skipping any columns with label(j)>0.
    end

    for j = i+1:last_j
        if label(j) > 0
            continue
        end

        if use_mex
            ip = ips(j-i);
        else
            if u(j) > ui/(1-radius)  % early termination 
                break
            end
            dist = dist + 1;
            ip = full(xi'*x(:,j)); % expensive
        end

        if (1-ip/(ui + u(j)-ip)) <= radius   % if tanimotodist <= radius
            label(j) = lab;
            gs(end) = gs(end) + 1;
        end
    end
    lab = lab + 1;
    waitbar(i/n,wb)
end
close(wb)
group_label = label; % store original group labels
out.t2_aggregate = toc(t);


%% merging %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t = tic;

gc_x = x(:,gc);
gc_u = u(gc);
gc_label = label(gc);  % will be [1,2,3,...]
A = spalloc(length(gc),length(gc),10*length(gc)); % adjacency of group centers

wb = waitbar(0,'CLASSIX\_T Merging');
for i = 1:length(gc)
    if ~merge_tiny_groups && gs(i) < minPts % tiny groups cannot take over large ones
        continue
    end

    xi = full(gc_x(:,i));      % current group center coordinate

    % get id = (tanimoto(xi,gc_x) <= merge_scale*radius); and igore id's < i
    if use_mex
        % TODO: could exploit that u is sorted (binary search)
        last_j = find(gc_u <= gc_u(i)/(1-merge_scale*radius), 1, 'last'); 

        if issparse(gc_x)
            ips = vecxspsubmat(xi,gc_x,i,last_j);
        else
            ips = matxsubmat(xi',gc_x,i,last_j);
        end
       
        ips = [ zeros(1,i-1) , ips , zeros(1,size(gc_x,2)-last_j) ];
        id = (1-ips./(gc_u(i) + gc_u - ips) <= merge_scale*radius);
        id(1:i-1) = 0;
    else
        ips = full(xi'*gc_x);
        id = (1-ips./(gc_u(i) + gc_u - ips) <= merge_scale*radius);
        id(1:i-1) = 0;
    end
        
    if ~merge_tiny_groups
        id = (id & (gs >= minPts));  % tiny groups are not merged into larger ones
    end
  
    A(id,i) = 1; % adjacency, keep track of merged groups 

    gcl = unique(gc_label(id)); % get all the affected group center labels
    % NOTE: gc_label(id) is not necessarily sorted!

    minlab = min(gcl);
    for L = gcl(:).'
        gc_label(gc_label==L) = minlab;   % important: need to relabel all of them,
    end                                   % not just the ones in id, as otherwise
                                          % groups that joined out of
                                          % order might stay disconnected
    waitbar(i/length(gc),wb)
end
close(wb)

% rename labels to be 1,2,3,... and determine cluster sizes
ul = unique(gc_label);
cs = zeros(length(ul),1);
for i = 1:length(ul)
    id = (gc_label==ul(i));
    gc_label(id) = i;
    cs(i) = sum(gs(id)); % cluster size = sum of all group sizes that form cluster
end

out.t3_merge = toc(t);

% At this point we have consecutive cluster gc_label (values 1,2,3,...)
% for each group center, and cs contains the total number of points
% for each cluster label.

%% minPts %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now eliminate tiny clusters by reassigning each of the constituting groups
% to the nearest group belonging to a cluster with at least minPts points. 
% This means, we are potentially dissolving tiny clusters, reassigning groups 
% to different clusters.

t = tic;

id = find(cs < minPts);   % cluster labels with small number of total points
copy_gc_label = gc_label; % added by Xinye (gc_label's before reassignment of tiny groups)

wb = waitbar(0,'CLASSIX\_T minPts');
for i = id(:)'
    ii = find(copy_gc_label==i); % find all tiny groups with that label
    for iii = ii(:)'
        xi = full(gc_x(:,iii));  % group center (starting point) of one tiny group
        
        ips = xi'*gc_x;
        d = 1 - ips./(gc_u(iii) + gc_u - ips);

        [~,o] = sort(d);      % indices of group centers ordered by distance from gc_x(:,iii)
        for j = o(:)'         % go through all of them in order and stop when a sufficiently large group has been found
            if cs(copy_gc_label(j)) >= minPts
                gc_label(iii) = copy_gc_label(j);
                break
            end
        end
    end
    waitbar(i/max(id),wb)
end
close(wb)

% rename labels to be 1,2,3,... and determine cluster sizes again
% needs to be redone because the tiny groups have now disappeared
ul = unique(gc_label);
cs = zeros(length(ul),1);
for i = 1:length(ul)
    id = (gc_label==ul(i));
    gc_label(id) = i;
    cs(i) = sum(gs(id));
end

% now relabel all labels, not just group centers
label = gc_label(label);

% unsort data labels
[~,J] = sort(ind);
label = label(J);
group_label = group_label(J);

% unsort group centers
gc = ind(gc); 
out.t4_minPts = toc(t);

%% explain function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t = tic;

% connectivity graph of group centers
G = graph(A,'lower');

% prepare explain handle
explain = @(varargin) explain_fun(varargin);

out.t5_finalize = toc(t);

%% prepare out structure
out.cs = cs;
out.dist = dist;
out.gc = gc;

    function explain_fun(varargin)
        args = varargin{1};

        if isempty(args) 
            fprintf('CLASSIX clustered %d data points with %d features.\n',size(x,2),size(x,1));
            fprintf('The radius parameter was set to %3.2f and minPts was set to %d.\n',radius,minPts); 
            fprintf('In total, %d distances were computed (%3.1f per data point).\n',dist,dist/size(x,2));
            fprintf('This resulted in %d groups, each with a unique group center.\n',length(gc));
            fprintf('These %d groups were subsequently merged into %d clusters.\n',length(gc),length(cs));
            fprintf('In order to explain the clustering of individual data points,\n');
            fprintf('use explain(ind1) or explain(ind1,ind2) with indices of points.\n');
            figure;
            
            if size(U,1) > 1e5
                fprintf('Too many data points for plot. Randomly subsampled 1e5 points.\n')
                plotind = randi(size(U,1),1e5,1);
                scatter(U(plotind,1),U(plotind,2),10,label(plotind),"filled");
            else
                scatter(U(:,1),U(:,2),10,label,"filled");
            end
            
            clim([0.95,length(cs)*1.05]); % make red (maxcolor) a bit brighter
            colormap jet
            hold on
            xlabel('total popcount')
            ylabel('odd popcount')
            title(sprintf("%d clusters (radius=%3.2f, minPts=%d)",length(cs),radius,minPts))
            axis tight equal
            return
        end

        if length(args) == 1 || (length(args)==2 && args{1}==args{2})
            ind1 = args{1};
            fprintf('Data point %d is in group %d, which was merged into cluster #%d.\n',ind1,group_label(ind1),label(ind1));
            figure; 
            if size(U,1) > 1e5
                fprintf('Too many data points for plot. Randomly subsampled 1e5 points.\n')
                plotind = randi(size(U,1),1e5,1);
                scatter(U(plotind,1),U(plotind,2),10,label(plotind),"filled",'MarkerFaceAlpha',.2);
            else
                scatter(U(:,1),U(:,2),10,label,"filled",'MarkerFaceAlpha',.2);
            end
            
            clim([0.95,length(cs)*1.05]); % make red (maxcolor) a bit brighter
            colormap jet
            hold on
            scatter(U(gc(group_label(ind1)),1),U(gc(group_label(ind1)),2),150,"g+",'LineWidth',3);
            scatter(U(ind1,1),U(ind1,2),150,"mx",'LineWidth',3);
            legend('scatter',sprintf('group center %d',group_label(ind1)),sprintf('data point %d (cluster #%d)',ind1,label(ind1)),'Location','southoutside','NumColumns',3);
            xlabel('total popcount')
            ylabel('odd popcount')
            title(sprintf("%d clusters (radius=%3.2f, minPts=%d)",length(cs),radius,minPts))
            axis tight equal
            %shg
            return
        end

        if length(args) == 2
            ind1 = args{1};
            ind2 = args{2};
            fprintf('Data point %d is in group %d, which was merged into cluster #%d.\n',ind1,group_label(ind1),label(ind1));
            fprintf('Data point %d is in group %d, which was merged into cluster #%d.\n',ind2,group_label(ind2),label(ind2));
            figure;

            if size(U,1) > 1e5
                fprintf('Too many data points for plot. Randomly subsampled 1e5 points.\n')
                plotind = randi(size(U,1),1e5,1);
                scatter(U(plotind,1),U(plotind,2),10,label(plotind),"filled",'MarkerFaceAlpha',.2);
            else
                scatter(U(:,1),U(:,2),10,label,"filled",'MarkerFaceAlpha',.2);
            end            
            clim([0.95,length(cs)*1.05]); % make red (maxcolor) a bit brighter
            colormap jet
            hold on
            scatter(U(gc(group_label(ind1)),1),U(gc(group_label(ind1)),2),150,"g+",'LineWidth',3);
            scatter(U(ind1,1),U(ind1,2),150,"mx",'LineWidth',3);
            scatter(U(gc(group_label(ind2)),1),U(gc(group_label(ind2)),2),150,"c+",'LineWidth',3);
            scatter(U(ind2,1),U(ind2,2),150,"mx",'LineWidth',3);
            
            if label(ind1)==label(ind2)   % if points in same cluster
                ii = find(label(gc)==label(ind1)); % also plot the group centers in that cluster
                ii = gc(ii);
                scatter(U(ii,1),U(ii,2),40,"k+","LineWidth",1,'MarkerEdgeAlpha',.4);
            
                % find and plot shortest path
                p = shortestpath(G,group_label(ind1),group_label(ind2));
                if ~isempty(p)
                    for g = p
                        scatter(U(gc(g),1),U(gc(g),2),100,"k+",'LineWidth',3);
                    end
                    fprintf('A path of overlapping groups with step size <= %3.2f*R = %3.2f is:\n',merge_scale,merge_scale*radius);
                    fprintf(' %d ->',p(1:end-1));
                    fprintf(' %d\n',p(end))
                else
                    fprintf('No path from group %d to group %d with step size <=%3.2f*R=%3.2f.\n',group_label(ind1),group_label(ind2),merge_scale,merge_scale*radius);
                    fprintf('This is because at least one of the groups was reassigned due\nto the minPts condition.\n');
                end
            else % different clusters
                fprintf('There is no path of overlapping groups between %d and %d.\n',group_label(ind1),group_label(ind2));
            end

            if label(ind1)==label(ind2)
                legend('scatter',sprintf('group center %d',group_label(ind1)),sprintf('data point %d (cluster #%d)',ind1,label(ind1)),sprintf('group center %d',group_label(ind2)),sprintf('data point %d (cluster #%d)',ind2,label(ind2)),sprintf('group centers in cluster #%d',label(ind1)),'Location','southoutside','NumColumns',2);
            else
                legend('scatter',sprintf('group center %d',group_label(ind1)),sprintf('data point %d (cluster #%d)',ind1,label(ind1)),sprintf('group center %d',group_label(ind2)),sprintf('data point %d (cluster #%d)',ind2,label(ind2)),'Location','southoutside','NumColumns',2);
            end
            xlabel('total popcount')
            ylabel('odd popcount')
            title(sprintf("%d clusters (radius=%3.2f, minPts=%d)",length(cs),radius,minPts))
            axis tight equal
            return
        end
    end

end

