#!/bin/env bash
folder=$1
if [ -z "$folder" ]
then
      folder="jobs"
fi
mkdir -p $1/running_jobs
mkdir -p $1/old_jobs
files=( $( ls -v $1/*.sh 2> /dev/null) )
while [ "${#files}" -ge 1 ]; do
    
   
    for f in ${files[@]}
    do 
 { # try
    nb_jobs=$(squeue -u samait -r -h -t pending,running | wc -l)
    if [ "${nb_jobs}" -le 999 ]; then
        echo "Submit try $f" && sbatch "$f" && mv "$f" "$1/running_jobs/" && echo "Successful submit" && continue
    else
        echo "Too many jobs in the queue, wait and recheck" && sleep 5 && break
    fi
    
    #save your output

} || { # catch
    {

     echo "Failed submit" && sleep 5 && echo "Resubmitted this very script again as job after sleeping 5 seconds" && break
} || { 
echo "Cannot resubmit this very script, sleep 5 seconds and retry..." && sleep 5 && break
}
}
break
done
files=( $( ls -v $1/*.sh 2> /dev/null) )
done

