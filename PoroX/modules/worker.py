import PoroX.modules.batch_utils as batch_utils

def collect_gameplay_experience(env, agent):
    """Collects gameplay experience from the environment using the agent.
    Currently very minimal to be used for testing purposes.
    """
    obs, infos = env.reset()
    
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}

    while not all(terminated.values()):
        batched_obs = batch_utils.collect_obs(obs)
        actions = agent.act(batched_obs)

        obs, rewards ,terminated, truncated, infos = env.step(actions)