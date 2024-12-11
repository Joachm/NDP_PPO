import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import realax as rx
import evosax as ex
import equinox as eqx
import equinox.nn as nn
from jaxtyping import PyTree
from typing import NamedTuple
from NDP_model import EdgeModel, DivModel, GeccoNDP, RNNModel, NDPToRNN
from PPO_model import *


class Config(NamedTuple):
    ##___PPO___##
    lr: float=5e-4
    gamma: float=0.99
    num_envs: int=4
    num_steps: int=128
    total_timesteps: int=int(5e5)
    update_epochs: int=4
    num_minibatches: int=4
    gae_lambda: float=0.95
    clip_eps: float=0.2
    ent_coef: float=0.001
    vf_coef: float=0.5
    max_grad_norm: float=0.5
    env_name: str="inverted_pendulum"
    anneal_lr: bool=False
    debug: bool=True
    ##__NDP__
    action_size:int=2
    input_size:int=4
    n_init_nodes:int=6
    max_nodes:int=35
    n_node_features:int=35
    max_dev_steps:int=1
    policy_iters:int=2

    @property
    def num_updates(self):
        return self.total_timesteps // self.num_steps // self.num_envs
    @property
    def minibatch_size(self):
        return self.num_envs * self.num_steps // self.num_minibatches


class ActorCritic_ndp(eqx.Module):
    # ---
    actor: RNNModel
    critic: nn.MLP
    sigma: jax.Array
    # ---
    def __init__(self, rnn_model,  obs_dims, action_dims,  *, key):
        k1, k2 = jr.split(key)
        self.actor = rnn_model  #RNNModel(weights, obs_dim, action_dims)
        self.critic = nn.MLP(obs_dims, "scalar",64, 2, key=jr.key(0), activation=jnn.tanh)

        self.sigma = jnp.zeros((1,))
    # ---
    def __call__(self, obs):
        h = jnp.zeros((self.actor.weights.shape[0],)) ##do not carry across timesteps
        mu = self.actor(obs, h)
        value = self.critic(obs)
        sigma = jnp.exp(self.sigma)
        pi = NormalDiag(mu, sigma)
        return pi, value

    # ---
    def initialize(self, key):
        return None

   
class MetaEvoTask:
    ##def __init__(self, inner_task, inner_prms_like,inner_steps=32, inner_pop=8, inner_es="SimpleGA"):
    def __init__(self, environment, inner_prms_like, inner_sttcs, config):
        self.env = environment
        self.inner_prms_like = inner_prms_like
        self.inner_sttcs = inner_sttcs
        self.cfg = config

    def __call__(self, outer_prms, key, data=None):

        #prms = eqx.combine(outer_prms, self.inner_prms_like)

        div_fn = DivModel(k1, n_node_features=cfg.n_node_features)
        edge_fn = EdgeModel(k2,
                            cfg.n_node_features)

        ndp = GeccoNDP(n_init_nodes=cfg.n_init_nodes,
                       n_node_features=cfg.n_node_features,
                       edge_fn=edge_fn,
                       max_nodes=cfg.max_nodes,
                       div_fn=div_fn,
                       )  # type: ignore


        p_like, statics = eqx.partition(ndp, eqx.is_array)
        ndp =  eqx.combine(outer_prms, statics)

        dev = NDPToRNN(ndp,
                         max_dev_steps=cfg.max_dev_steps,
                         action_dims=cfg.action_size,
                         obs_dims=cfg.input_size)

        policy_state, _ = dev.initialize(k2)
        rnn_model = dev.create_rnn(policy_state, cfg.policy_iters)

        mdl = ActorCritic_ndp(rnn_model, cfg.input_size, cfg.action_size, key=k1)
        inner_prms, _ = eqx.partition(mdl, eqx.is_array)

        rl_state, [losses, scores]= self.inner_train(inner_prms, key)
        return scores.mean(), [losses, scores]
        #return scores[-1], [losses, scores]

    def inner_train(self, prms, key):
        trainer = PPO(lambda p: eqx.combine(p, self.inner_sttcs), self.env, self.cfg)
        return trainer.train(prms, k2)



key = jr.key(0)
k1, k2 = jr.split(key)


env_name = 'ant'
env  = create(env_name, episode_length=1000, backend='mjx')

action_size = env.action_size
input_size = env.observation_size
max_nodes = 35#config["model_config"]["model_params"]["num_hidden_neurons"] + action_size + input_size + 1



cfg = Config(
gamma = 0.99,
gae_lambda = 0.95,
ent_coef = 0.,
vf_coef = 1.,
update_epochs=1,
max_grad_norm = 0.6,
clip_eps = 0.1,
num_envs=16,
num_minibatches=32,
num_steps=256,
lr=3e-4,
total_timesteps=int(6e3),
env_name=env_name,
anneal_lr=False,
##___NPD_PARAMS___
action_size=action_size,
input_size=input_size,
n_init_nodes=action_size + input_size + 1,   ##+1 for bias
max_nodes=max_nodes,
n_node_features=max_nodes,
max_dev_steps=15,
policy_iters=3,
)


# init trainable npd modules
div_fn = DivModel(k1, n_node_features=cfg.n_node_features)
edge_fn = EdgeModel(k2,
                    cfg.n_node_features)

ndp = GeccoNDP(n_init_nodes=cfg.n_init_nodes,
               n_node_features=cfg.n_node_features,
               edge_fn=edge_fn,
               max_nodes=cfg.max_nodes,
               div_fn=div_fn,
               )  # type: ignore



dev = NDPToRNN(ndp,
                 max_dev_steps=cfg.max_dev_steps,
                 action_dims=cfg.action_size,
                 obs_dims=cfg.input_size)

policy_state, _ = dev.initialize(k2)
rnn_model = dev.create_rnn(policy_state, cfg.policy_iters)

##initialize the actor-critic with a random rnn_model
mdl = ActorCritic_ndp(rnn_model, cfg.input_size, cfg.action_size, key=k1)
inner_prms, inner_sttcs = eqx.partition(mdl, eqx.is_array)


tsk = MetaEvoTask(env, inner_prms,inner_sttcs, cfg)

logger = rx.Logger(
    wandb_log=True, # if data should be logged to wandb
    metrics_fn=rx.logging.log.default_es_metrics, # will log min, max and mean firness and ckpt current es mean
    ckpt_file="ndp_test_ant_des",
    ckpt_freq=50)

config = {
    "env": cfg.env_name
}

logger.init(project="jax_npd_ppo_", config=config)

outer_prms, outer_sttcs = eqx.partition(ndp, eqx.is_array)
trainer = rx.EvosaxTrainer(10000, "DES", tsk, outer_prms, popsize=256, fitness_shaper=ex.FitnessShaper(maximize=True), logger=logger)

print('begin training')
final_state = trainer.init_and_train_(jr.key(1))

logger.finish()


