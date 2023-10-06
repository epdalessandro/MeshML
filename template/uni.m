function uni

N = 8
xv = 0.3 + 1.5*linspace(0,1,4);
yv = 0.3 + 0.4*linspace(0,1,2);
[Y,X] = meshgrid(yv, xv);
XY = [reshape(X,N,1), reshape(Y,N,1)];

unix('rm -f ../runs/all.run');

for n = 1:size(XY,1)
  setup_run(XY(n,1), XY(n,2), sprintf('run_%d', n));
end
fid = fopen('XY.txt', 'w');
fprintf(fid, '%.15e %.15e\n', XY');
fclose(fid);


