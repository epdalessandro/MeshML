function setup_run(pulsex, pulsey, froot);
% gustV = gust velocity
% gustH = gust height
% froot = root name for writing the files

% where to put the files
spath = sprintf('../runs');
unix(sprintf('mkdir -p %s', spath));
    
% key-value pairs to replace in .job file
KVjob = {'SAVEPREFIX', froot;
         'EQNSETFILE', sprintf('%s.eqn', froot);
        };

% sed job file -> place in directory
fjob = sprintf('%s/%s.job', spath, froot);
WriteFromTemplate('template.job', KVjob, fjob);
    
% key-value pairs to replace in .eqn file
KVeqn = {'PULSEX', sprintf('%.15e', pulsex);
         'PULSEY', sprintf('%.15e', pulsey);
        };

% sed eqn files -> place in directory
WriteFromTemplate('template.eqn', KVeqn, ... 
                  sprintf('%s/%s.eqn',spath, froot));

    
% run file -> place in directory
frun = sprintf('%s/all.run', spath);
fid = fopen(frun, 'a');
fprintf(fid, 'nohup xflow %s.job > %s.txt &\n', froot, froot);
fclose(fid);


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
  



