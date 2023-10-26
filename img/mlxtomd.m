
fn = 'README';
mlx = [ fn '.mlx' ];
md = [ fn '.md' ];

mdfile = export(mlx,Format="markdown",EmbedImages=0);

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
f = strrep(f,'[MATLABONLINEBADGE]','[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=nla-group/classix-matlab&file=README.mlx)');
f = strrep(f,'[FILEEXCHANGEBADGE]','[![View CLASSIX: Fast and explainable clustering on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://uk.mathworks.com/matlabcentral/fileexchange/153451-classix-fast-and-explainable-clustering)');
fid = fopen(md,'w');
fprintf(fid,'%s',f);
fclose(fid);