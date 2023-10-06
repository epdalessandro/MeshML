function lhs

% viscosity and alpha
rng default;
nrun = 100;
X = lhsdesign(nrun, 3);

% viscosities
mu = 5*10.^(-7+X(:,1));

% angles of attack (0 -> 5 deg)
alpha = X(:,2)*5*pi/180.;
ca = cos(alpha); sa = sin(alpha);

% mach number
mach = 0.1+0.9*X(:,3); % TODO: cases going supersonic/transonic so look into defaults for lhs and see if I need to set some limits on mach range like for angle

fid = fopen('param.txt', 'w');
fprintf(fid, '%.15e %.15e %.15e %.15e\n', [mu'; ca'; sa'; mach']);
fclose(fid);
