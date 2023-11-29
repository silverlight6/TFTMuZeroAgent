def collect_gameplay_experience(env, agent):
    """Collects gameplay experience from the environment using the agent.
    Currently very minimal to be used for testing purposes.
    """
    obs, infos = env.reset()
    
    print(obs["player_0"]["player"]["scalars"])
    return
    
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}

    while not all(terminated.values()):
        actions = agent.act(obs)

        obs, rewards ,terminated, truncated, infos = env.step(actions)