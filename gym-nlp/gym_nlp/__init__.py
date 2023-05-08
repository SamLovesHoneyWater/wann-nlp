from gym.envs.registration import register

register(
    id='nlp-v0',
    entry_point='gym_nlp.envs:NLPEnv_v0',
)

register(
    id='nlp-v1',
    entry_point='gym_nlp.envs:NLPEnv_v1',
)