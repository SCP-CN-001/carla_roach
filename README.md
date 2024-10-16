# Roach

This repository provides a migrated version of Roach for CARLA Leaderboard 2.0, serving as part of the benchmark suite. The implementation has referred to the [ThinkTwice](https://github.com/OpenDriveLab/DriveAdapter/tree/main) codebase used for Leaderboard 1.0.

## Evaluate on CARLA Leaderboard 1.0

### Setup the environment

Create soft link to the CARLA simulator folder.

```shell
ln -s /path/to/carla-simulator-0.9.10.1 ./CARLA_Leaderboard_10
```

Pull submodules for the leaderboard.

```shell
git submodule update --init --recursive
```

### Run the evaluation

```shell
cd scripts
./L10_run_evaluation.sh
```

To avoid unexpected stop due to Carla Simulator, we recommend to run the evaluation route by route. You can change the route index by modifying `ROUTES_SUBSET` in `L10_run_evaluation.sh`.

## Run the LeaderboardV2_Roach

```shell
cd leaderboard
# edit the LEADERBOARD_ROOT path and Carla_root path
bash run_lv2_roach.sh
```

## Thanks to

- [ThinkTwice](https://github.com/OpenDriveLab/DriveAdapter/tree/main)'s implementation version.
- [Roach](https://github.com/zhejz/carla-roach)'s original implementation version.

## Citation

```latex
```
