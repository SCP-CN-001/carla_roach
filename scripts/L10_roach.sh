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
export route_name=routes_town01_00.xml
export ROUTES=${LEADERBOARD_ROOT}/data/routes_devtest.xml
export TEAM_AGENT=${WORKSPACE}/roach/roach_agent.py
export TEAM_CONFIG=${WORKSPACE}/roach/configs/roach_agent.yaml
export TME_STAMP=$(date +"%Y_%m_%d_%H_%M_%s")
export SCENARIOS=${LEADERBOARD_ROOT}/data/all_towns_traffic_scenarios_public.json
export ROUTE_FILE=None
export CHECKPOINT_ENDPOINT=lv1_roach_result.json

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${WORKSPACE}/logs/L10/${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT} \
--is_eval=True
