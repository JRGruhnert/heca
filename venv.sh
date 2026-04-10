export HOOPGN_ROOT=/home/jangruhnert/Documents/GitHub/master-project
export CALVIN_ENV_ROOT=$HOOPGN_ROOT/external/calvin_env_modified
export TAPAS_ROOT=$HOOPGN_ROOT/external/tapas_gmm_modified
export PYTHONPATH=$HOOPGN_ROOT:$CALVIN_ENV_ROOT:$TAPAS_ROOT:$PYTHONPATH

echo "Paths set!"
echo "HOOPGN_ROOT: $HOOPGN_ROOT"
echo "PYTHONPATH: $PYTHONPATH"