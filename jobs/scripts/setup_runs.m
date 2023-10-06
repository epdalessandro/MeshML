function setup_runs;

% Globals
Base = 'naca'; % base name for files
AdaptOutput = 'Drag';

% target cost in dof
TargetCost = 30000;

% order for solutions
p = 2;
POrder = sprintf('%d', p);

% constant parameters
rho=1; u=1;
SA_NonDim = 1e-3;

% load run parameters
sparam = {'VISCOSITY', 'COSALPHA', 'SINALPHA', 'MACHNUMBER'};
vparam = load('param.txt');
nrun = size(vparam,1);

% make xfa directory: this is where the xfa files will go
unix('mkdir -p ../xfa');

% run and harvest scripts
unix('rm -rf all.run all.hvs');

% loop over runs
for irun = 1:nrun,

  % calculate variables
  M = vparam(irun,4); % mach number
  c=u/M; press=c^2*rho/1.4;
  TotEnergy = press/0.4 + 0.5*rho*u*u;

  % make run directory
  spath = sprintf('../runs/run-%d', irun);
  unix(sprintf('mkdir -p %s', spath));
  
  DORESTART = 'False';
  INPUTFILE = '..\/meshes\/naca.gri';
  GEOMFILE = '..\/meshes\/naca.geom';
  INTERPORDER = POrder;
  SEQUENCEORDERADD = '0';

  % key-value pairs to replace in .job file
  KVjob = {'DORESTART', DORESTART;
           'INPUTFILE', INPUTFILE;
           'INTERPORDER', INTERPORDER;
           'SEQUENCEORDERADD', SEQUENCEORDERADD;
           'GEOMFILE', GEOMFILE;
           'ADAPTOUTPUT', AdaptOutput;
           'SAVEPREFIX', Base;
           'TARGETCOST', sprintf('%.0f',TargetCost);
          };
    
  % sed job file -> place in directory
  fjob = sprintf('%s/naca.job', spath);
  WriteFromTemplate('template.job', KVjob, fjob);
  
  % key-value pairs to replace in .eqn file
  KVeqn = {'VISCOSITY', sprintf('%.15e', vparam(irun,1));
           'COSALPHA', sprintf('%.15e', vparam(irun,2));
           'SINALPHA', sprintf('%.15e', vparam(irun,3));
           'MACHNUMBER', sprintf('%.15e', vparam(irun,4));
           'TOTENERGY', sprintf('%.15e', TotEnergy);
           'NUTILINF', sprintf('%.15e', 3*vparam(irun,1)/SA_NonDim);
           'SA_NONDIM', sprintf('%.15e', SA_NonDim);
          };

  % sed eqn file -> place in directory
  feqn = sprintf('%s/naca.eqn', spath);
  WriteFromTemplate('template.eqn', KVeqn, feqn);

  % link bamg to run directory
  unix(sprintf('(cd %s; ln -s ../meshes/bamg .)', spath)); 
  
  % append to run and harvest files
  fid = fopen('all.run', 'a');
  fprintf(fid, '(cd %s; nohup ../../../../bin/xflow naca.job > naca.txt &)\n', spath);
  fclose(fid);
  fid = fopen('all.hvs', 'a');
  fprintf(fid, '(cd %s; cp naca_0.xfa ../../xfa/run-%d.xfa; cd ../../xfa; ../../../bin/xf_Convert -in run-%d.xfa -out run-%d-unref.gri)\n', spath, irun, irun, irun);
  fclose(fid);
  
end

% make executable
unix('chmod 755 all.run all.hvs');

%-------------------------------------------------------------
function WriteFromTemplate(ftemplate, KV, fdest);

sedstr = sprintf('cat %s', ftemplate);
for i = 1:size(KV,1),
  key = KV{i,1};
  val = KV{i,2};
  sedstr = sprintf('%s | sed "s/%s/%s/g"', sedstr, key, val);
end
sedstr = sprintf('%s > %s', sedstr, fdest);
unix(sedstr);
  

