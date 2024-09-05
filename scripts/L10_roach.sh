# setup environment variables
export SHELL_PATH=$(dirname $(readlink -f $0))
export WORKSPACE=${SHELL_PATH}/..

export CARLA_ROOT=${WORKSPACE}/CARLA_Leaderboard_10
export LEADERBOARD_ROOT=${WORKSPACE}/leaderboard_10/leaderboard
export SCENARIO_RUNNER_ROOT=${WORKSPACE}/leaderboard_10/scenario_runner

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}
export PYTHONPATH=$PYTHONPATH:${SCENARIO_RUNNER_ROOT}
export PYTHONPATH=$PYTHONPATH:${WORKSPACE}
export PYTHONPATH=$PYTHONPATH:${WORKSPACE}/leaderboard/team_code

export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2000
export TM_PORT=2500
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export RESUME=False

# Roach data collection
export ROUTES=${LEADERBOARD_ROOT}/data/routes_testing.xml
export ROUTES_SUBSET=0
export SCENARIOS=${LEADERBOARD_ROOT}/data/all_towns_traffic_scenarios_public.json

export TEAM_AGENT=${WORKSPACE}/roach/roach_agent.py
export TEAM_CONFIG=${WORKSPACE}/roach/configs/roach_agent.yaml

time_stamp=$(date +"%Y_%m_%d_%H_%M_%s")
export CHECKPOINT_ENDPOINT=${WORKSPACE}/logs/L10_testing/log_route_${ROUTES_SUBSET}_${time_stamp}.json

python3 ${WORKSPACE}/roach/leaderboard_custom/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--routes-subset=${ROUTES_SUBSET} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT} \
--is_eval=True
