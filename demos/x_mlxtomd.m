
%export("Using_the_CLASSIX_Python_package_in_MATLAB.mlx","Using_the_CLASSIX_Python_package_in_MATLAB.ipynb")


fn = 'Comparing_MATLAB_and_Python_runtimes';
mlx = [ fn '.mlx' ];
md = [ fn '.md' ];

mdfile = export(mlx,Format="markdown",EmbedImages=0);
%%
pause(2)

fld0 = [ fn '_media' ];
fld1 = [ 'img\' fld0 ];

try
    system(['rmdir ' fld1 ' /s /q']);
catch
end
copyfile(fld0,fld1); % copy media folder into img
try
    system(['rmdir ' fld0 ' /s /q']);
catch
end

%%

fid = fopen(md,'r');
f = fread(fid,'*char')';
fclose(fid);
fld1 = strrep(fld1,'\','/'); % needed for Matlab FileExc link
f = strrep(f,fld0,fld1);
fid = fopen(md,'w');
fprintf(fid,'%s',f);
fclose(fid);