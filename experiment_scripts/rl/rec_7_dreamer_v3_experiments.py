from gym import make, register
import dill
import crafter


import warnings
import dreamerv3
from dreamerv3 import embodied
from embodied.envs import from_gym
from dreamerv3.train import make_replay
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

def main():

    
    rec_envs = dill.load(open("rec_7_envs.env", "rb"))
    env_train_id, env_train = rec_envs["env_train"]
    env_eval_id, env_eval = rec_envs["env_eval"]
    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['medium'])
    #config = config.update(dreamerv3.configs['multicpu'])
    config = config.update({
        'envs.amount': 1,
        'run.train_ratio': 512,
        'run.eval_every': 100,
        'run.log_every': 60,
        'batch_size': 32,
        "batch_length": 64,
        'jax.prealloc': False,
        'encoder.cnn_keys': '$^',
        'decoder.cnn_keys': '$^',
        'encoder.mlp_keys': 'vector',
        'decoder.mlp_keys': 'vector',
        'jax.platform': 'cpu',
        #'rssm.units': 256,
        "jax.precision": "float32",
        "horizon": env_train.T,
        "run.eval_initial": False,
        "run.eval_eps":4,
        "rssm.unroll": False,
        "run.train_fill": 10*env_train.T,
        "run.expl_until": 0,
        "retnorm.decay": 0.9999
    })
    config = embodied.Flags(config).parse()

    logdir = None
    step = embodied.Counter()
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        # embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        # embodied.logger.TensorBoardOutput(logdir),
        # embodied.logger.WandBOutput(logdir.name, config),
        # embodied.logger.MLFlowOutput(logdir.name),
    ])

    
    
    

    env = from_gym.FromGym(env_train, obs_key='vector')  # Or obs_key='vector'.
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    eval_env = from_gym.FromGym(env_eval, obs_key='vector')  # Or obs_key='vector'.
    eval_env = dreamerv3.wrap_env(eval_env, config)
    eval_env = embodied.BatchEnv([eval_env], parallel=False)
    """
    env = make("Pendulum-v1")#crafter.Env()  # Replace this with your Gym env.
    env = from_gym.FromGym(env, obs_key='vector')  # Or obs_key='vector'.
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)
    eval_env=env
    """
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = make_replay(config, logdir)
    eval_replay = make_replay(config, logdir, is_eval=True)

    args = embodied.Config(
        **config.run, logdir=logdir,
        batch_steps=config.batch_size * config.batch_length)
    embodied.run.train_eval(agent, env, eval_env, replay, eval_replay, logger, args)
    # embodied.run.eval_only(agent, env, logger, args)

    """
    # See configs.yaml for all options.
            config = embodied.Config(dreamerv3.configs['defaults'])
            config = config.update(dreamerv3.configs['medium'])
            config = config.update({
                'logdir': '~/logdir/run1',
                'run.train_ratio': 64,
                'run.log_every': 30,  # Seconds
                'batch_size': 16,
                'jax.prealloc': False,
                'encoder.mlp_keys': '$^',
                'decoder.mlp_keys': '$^',
                'encoder.cnn_keys': 'image',
                'decoder.cnn_keys': 'image',
                'jax.platform': 'cpu',
            })
            config = embodied.Flags(config).parse()

            logdir = embodied.Path(config.logdir)
            step = embodied.Counter()
            logger = embodied.Logger(step, [
                embodied.logger.TerminalOutput(),
                embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
                embodied.logger.TensorBoardOutput(logdir),
                # embodied.logger.WandBOutput(logdir.name, config),
                # embodied.logger.MLFlowOutput(logdir.name),
            ])

            from embodied.envs import from_gym
            import gym
            env = gym.make(env_train_id)
            env = from_gym.FromGym(env, obs_key='vector')  # Or obs_key='vector'.
            env = dreamerv3.wrap_env(env, config)
            env = embodied.BatchEnv([env], parallel=False)

            eval_env = gym.make(env_eval_id)
            eval_env = from_gym.FromGym(eval_env, obs_key='vector')  # Or obs_key='vector'.
            eval_env = dreamerv3.wrap_env(eval_env, config)
            eval_env = embodied.BatchEnv([eval_env], parallel=False)

            replay = make_replay(config, logdir / 'replay')
            eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)

            agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
            replay = embodied.replay.Uniform(
                config.batch_length, config.replay_size, logdir / 'replay')
            args = embodied.Config(
                **config.run, logdir=config.logdir,
                batch_steps=config.batch_size * config.batch_length)
            embodied.run.train_eval(agent, env, eval_env, replay, eval_replay, logger, args)
            # embodied.run.eval_only(agent, env, logger, args)

    """


if __name__ == '__main__':
  main()
