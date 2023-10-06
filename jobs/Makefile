# work server + path which to rsync
WORK = epdal@pmultigrid.engin.umich.edu:/home/epdal/xflow/mesh_generation/jobs

# file listing which directories/files to exclude
EXCLUDE = exclude.txt

# all is sync
all: sync

# newer server files gotten from server
get:
	rsync -avzu '$(WORK)' --exclude-from $(EXCLUDE) .

# newer local files put onto server
put:
	rsync -avzu * --exclude-from $(EXCLUDE) $(WORK)

# synchronize by getting and putting
sync: get put

