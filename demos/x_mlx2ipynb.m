% convert all mlx files to ipynb

clear all
f = dir('*.mlx'); 
for i = 1:length(f)
    fn1 = f(i).name;
    fn2 = strrep(fn1,'.mlx','.ipynb');
    disp(['Converting ' fn1])
    export(fn1,fn2);
end
disp('All done.')