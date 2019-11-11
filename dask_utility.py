import sys
import socket
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, wait

#dask generate utiltiies
def start_tunnel():
    """
    Start a juypter session and ssh tunnel to view task progress
    """
    host = socket.gethostname()        
    print("To tunnel into dask dashboard:")
    print("ssh -N -L 8787:%s:8787 -l b.weinstein hpg2.rc.ufl.edu" % (host))
    
    #flush system
    sys.stdout.flush()
    
def start_dask_cluster(number_of_workers, mem_size="10GB"):

    #################
    # Setup dask cluster
    #################

    #job args
    extra_args=[
        "--error=/home/b.weinstein/logs/dask-worker-%j.err",
        "--account=ewhite",
        "--output=/home/b.weinstein/logs/dask-worker-%j.out"
    ]

    cluster = SLURMCluster(
        processes=1,
        queue='hpg2-compute',
        cores=1, 
        memory=mem_size, 
        walltime='24:00:00',
        job_extra=extra_args,
        local_directory="/home/b.weinstein/logs/dask/", death_timeout=300)

    print(cluster.job_script())
    cluster.adapt(minimum=number_of_workers, maximum=number_of_workers)

    dask_client = Client(cluster)

    #Start dask
    dask_client.run_on_scheduler(start_tunnel)  
    
    return dask_client
