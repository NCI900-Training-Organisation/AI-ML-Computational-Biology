#!/bin/bash
# adopted from /apps/pangeo/2021.01

#PBS_O_WORKDIR: The directory your job was submitted from
if [ -e ${PBS_O_WORKDIR}/client_cmd  ]; then
    rm  ${PBS_O_WORKDIR}/client_cmd
fi

# no need to check occupied port, as jupyter could pick up a spare one.
jport=$(shuf -n 1 -i 8301-8400)

# write ssh command to client_cmd
echo "ssh -N -L ${jport}:${HOSTNAME}:${jport} ${USER}@gadi.nci.org.au ">& client_cmd
echo "Jupyter lab started ..."
echo "client_cmd created ..."
echo `hostname`
jupyter lab --no-browser --ip=`hostname` --port=${jport}




