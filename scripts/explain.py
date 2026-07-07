from dataclasses import dataclass
from omegaconf import OmegaConf, SCMode

from tapas_gmm.utils.argparse import parse_and_build_config

from scripts.train import TrainConfig
from src.agents.ppo.gnn import GNNAgent
from src.modules.buffer import Buffer
from src.modules.storage import Storage
from src.factory import (
    select_agent,
    select_environment,
    select_experiment,
    select_evaluator,
)


@dataclass
class ExplainConfig:
    bb: TrainConfig
    bp: TrainConfig
    br: TrainConfig


def run_explain(config: ExplainConfig):
    bb_storage = Storage(config.bb.storage)
    bb_buffer = Buffer(config.bb.buffer)
    bb_evaluator = select_evaluator(config.bb.evaluator, bb_storage)
    bb_env = select_environment(config.bb.environment, bb_evaluator, bb_storage)
    bb_experiment = select_experiment(config.bb.experiment, bb_env, bb_storage)
    bb_agent = select_agent(config.bb.agent, bb_storage, bb_buffer)

    assert isinstance(bb_agent, GNNAgent), "Agent must be GNNAgent for explanation"

    obs, goal = bb_experiment.sample_task()
    state_diff = bb_evaluator.is_equal_dict(obs, goal)
    while state_diff["block_blue_position"]:
        obs, goal = bb_experiment.sample_task()
        state_diff = bb_evaluator.is_equal_dict(obs, goal)
    ref_obs = obs
    ref_goal = goal

    print(state_diff)
    step = 0
    skills_list = []
    done = False
    episode_ended = False
    while not episode_ended:
        skill = bb_agent.explain(obs, goal, step)
        # print(f"[step {step}] skill = {skill.name}")
        skills_list.append(skill.name)
        obs, reward, done, episode_ended = bb_experiment.step(skill)
        bb_agent.feedback(reward, done, episode_ended)
        step += 1
    bb_experiment.close()

    print(f"Executed skills: {skills_list}")
    print(f"Done: {done}, episode_ended: {episode_ended}")

    bp_storage = Storage(config.bp.storage)
    bp_buffer = Buffer(config.bp.buffer)
    bp_evaluator = select_evaluator(config.bp.evaluator, bp_storage)
    bp_env = select_environment(config.bp.environment, bp_evaluator, bp_storage)
    bp_experiment = select_experiment(config.bp.experiment, bp_env, bp_storage)
    bp_agent = select_agent(config.bp.agent, bp_storage, bp_buffer)
    assert isinstance(bp_agent, GNNAgent), "Agent must be GNNAgent for explanation"

    bp_obs, bp_goal = bp_experiment.sample_task()
    while (
        not bp_evaluator.is_equal(ref_obs, bp_obs)
        or not bp_evaluator.is_equal(ref_goal, bp_goal)
        or not bp_evaluator.is_equal_dict(bp_obs, bp_goal) == state_diff
    ):
        bp_obs, bp_goal = bp_experiment.sample_task()

    step = 0
    skills_list = []
    done = False
    episode_ended = False
    while not episode_ended:
        skill = bp_agent.explain(bp_obs, bp_goal, step)
        # print(f"[step {step}] skill = {skill.name}")
        skills_list.append(skill.name)
        bp_obs, reward, done, episode_ended = bp_experiment.step(skill)
        bp_agent.feedback(reward, done, episode_ended)
        step += 1
    bp_experiment.close()

    print(f"Executed skills: {skills_list}")
    print(f"Done: {done}, episode_ended: {episode_ended}")

    br_storage = Storage(config.br.storage)
    br_buffer = Buffer(config.br.buffer)
    br_evaluator = select_evaluator(config.br.evaluator, br_storage)
    br_env = select_environment(config.br.environment, br_evaluator, br_storage)
    br_experiment = select_experiment(config.br.experiment, br_env, br_storage)
    br_agent = select_agent(config.br.agent, br_storage, br_buffer)
    assert isinstance(br_agent, GNNAgent), "Agent must be GNNAgent for explanation"

    br_obs, br_goal = br_experiment.sample_task()
    while (
        not br_evaluator.is_equal(ref_obs, br_obs)
        or not br_evaluator.is_equal(ref_goal, br_goal)
        or not br_evaluator.is_equal_dict(br_obs, br_goal) == state_diff
    ):
        br_obs, br_goal = br_experiment.sample_task()

    step = 0
    skills_list = []
    done = False
    episode_ended = False
    while not episode_ended:
        skill = br_agent.explain(br_obs, br_goal, step)
        # print(f"[step {step}] skill = {skill.name}")
        skills_list.append(skill.name)
        br_obs, reward, done, episode_ended = br_experiment.step(skill)
        br_agent.feedback(reward, done, episode_ended)
        step += 1
    br_experiment.close()

    print(f"Executed skills: {skills_list}")
    print(f"Done: {done}, episode_ended: {episode_ended}")


def entry_point():
    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )
    run_explain(config)  # type: ignore


if __name__ == "__main__":
    entry_point()
