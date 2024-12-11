from jaxtyping import Array, Int, PyTree
from typing import Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox.nn as nn
import equinox as eqx
from typing import Mapping, NamedTuple, Tuple
import numpy as onp

from ndp import BaseNDP, Graph, State, Node
from functools import partial
from jaxtyping import Float, Array, PyTree, Int, Bool
from flax import linen


class Node(NamedTuple):
    mask: Optional[Float[Array, "N"]] = None
    embedding: Optional[Float[Array, "N"]] = None
    p: Optional[Float[Array, "N X"]] = None
    pholder: Optional[PyTree] = None


class Edge(NamedTuple):
    adj: Optional[Float[Array, "..."]] = None
    weights: Optional[Float[Array, "E De"]] = None


class PolicyState(NamedTuple):
    adj: jax.Array  # adjacency matrix
    weights: jax.Array  # weight matrix
    mask: jax.Array  # currently alive nodes
    rnn_state: jax.Array  # hidden state inherited between steps of an episode
    node_embedding: jax.Array


class EdgeModel(eqx.Module):
    """ Assigns a weight to an edge
    """
    mlp: PyTree
    n_node_features: int
    normalize_input: bool


    def __init__(self, key, n_node_features, normalize_input=True):

        key_mp, key_mlp = jr.split(key)
        self.n_node_features = n_node_features
        self.normalize_input = normalize_input

        self.mlp =  nn.MLP(self.n_node_features*2,
                         1,
                         16, 2, key=key_mlp, activation=linen.leaky_relu,
                         final_activation=lambda x: x, use_bias=False, use_final_bias=False)


    def __call__(self, graph, key):
        node_types = graph.nodes.embedding
        n = node_types.shape[0]
        pre = jnp.repeat(node_types[:, None], n, axis=1)
        post = jnp.repeat(node_types[None, :], n, axis=0)
        input = jnp.concatenate([pre, post], axis=-1)

        w = jax.vmap(jax.vmap(self.mlp))(input)
        w = jnp.squeeze(w,axis=-1)
        #w = jnp.where(graph.edges.adj, w, jnp.zeros_like(w))

        return w

class DivModel(eqx.Module):
    mlp: nn.MLP
    n_node_features: int
    normalize_input: bool

    def __init__(self, key, n_node_features, normalize_input=True):
        self.n_node_features = n_node_features
        self.normalize_input = normalize_input
        self.mlp = nn.MLP(self.n_node_features, 1, 32, 1, activation=jnn.relu, final_activation=jax.nn.tanh, key=key)

    def __call__(self, graph, key):
        node_types = graph.nodes.embedding
        d = jax.vmap(self.mlp)(node_types)
        d = (d>0.).astype(float)
        return d

class GeccoNDP(BaseNDP):
    """
    """

    def __init__(self, edge_fn, div_fn, n_node_features,  max_nodes, n_init_nodes
                 ):
        super().__init__(edge_fn=edge_fn,
                         node_fn=None,
                         div_fn=div_fn,
                         max_nodes=max_nodes,
                         n_node_features=n_node_features,
                         n_init_nodes=n_init_nodes)

    # -------------------------------------------------------------------

    def __call__(self, graph: Graph, key: jax.Array, counter: int, dev_step, index,
                 target_function) -> Graph:
        graph = self.add_new_nodes(graph,
                                   key, counter=counter,
                                   dev_step=dev_step,
                                   index=index,
                                   target_function=target_function)
        graph = self.update_edges(graph, key)
        return graph

    def add_new_nodes(self, graph: Graph, key: jax.Array, counter: int, dev_step: int, index: int,
                      target_function: int) -> Graph:
        """"""

        # each cell decides whether to grow another cell
        grow = jnp.squeeze(self.div_fn(graph, key))
        alive = graph.nodes.mask
        grow = grow * alive

        #grow = grow.at[0].set(0)
        n_grow = grow.sum()
        N = grow.shape[0]
        n_alive = alive.sum()
        new_n = n_alive + n_grow
        new_mask = (jnp.arange(graph.edges.adj.shape[0]) < new_n).astype(float) # who is alive now?
        xnew_mask = new_mask - graph.nodes.mask # who did come alive now?

        # compute childs index for each parent: pc[parent id] = child id
        pc = (jnp.where(grow, jnp.cumsum(grow) - 1, -1) + (n_alive * grow)).astype(int)

        # Set child´s incoming connections (parent neighbors + parents)
        nA = jax.ops.segment_sum(jnp.identity(N), pc, N).T
        adj = jnp.where(xnew_mask[None, :], nA, graph.edges.adj) * new_mask[None, :] * new_mask[:, None]


        # update node embeddings
        new_embeddings = self.update_embeddings(graph.nodes.embedding, grow, pc,
                                                               key)

        nodes = graph.nodes._replace( mask=new_mask.astype(float),
                                     embedding=new_embeddings)

        return graph._replace(nodes=nodes, edges=graph.edges._replace(adj=adj))

    def rollout(self, state: State, key: jr.PRNGKey, max_steps, target_function) -> Tuple[
        State, State]:
        def _step(c, x):
            s, k,  counter, dev_step, index = c
            counter += 1
            k, k_ = jr.split(k)

            s = self.__call__(s, k_, counter, dev_step, index,  target_function)
            #
            # s = self.__call__(s, k_, counter, dev_step, index, current_key_fixed, target_function)
            index += dev_step
            dev_step *= 2
            return [s, k, counter, dev_step, index], s

        [state, _,  counter, _, _], states = jax.lax.scan(
            _step, [state, key, 0, 1, 0], None, max_steps
        )

        # state = self.update_edges(state, key)
        return state, states

    def init_embeddings(self, adj, key):
        """ Input and output nodes get a unique ID that will be inherited to grown nodes
        """
        unique_embeddings = jnp.where(jnp.arange(self.max_nodes) < self.n_init_nodes, jnp.arange(self.max_nodes), 0)
        unique_embeddings = jax.nn.one_hot(unique_embeddings, num_classes=self.max_nodes)
        return unique_embeddings

    def update_embeddings(self, embeddings,  grow, pc, key):
        # inherit unique embedding from parent
        children = jnp.where(grow, pc, -100)
        parent_intrinsic = embeddings[..., :]
        noise = jax.random.normal(key, parent_intrinsic.shape) * 0.1
        noise = 0
        embeddings = embeddings.at[children, ...].set(parent_intrinsic + noise)

        return embeddings

    def initialize(self, key):
        init_adj = jnp.zeros((self.max_nodes, self.max_nodes))
        init_adj = init_adj.at[:self.n_init_nodes, :self.n_init_nodes].set(1)
        init_emb = self.init_embeddings(init_adj, key)

        nodes = Node(
            embedding=init_emb,
            mask=jnp.zeros((self.max_nodes,)).at[:self.n_init_nodes].set(1.))

        edges = Edge(
            adj=init_adj,
            weights=jr.normal(key, (self.max_nodes, self.max_nodes)) * jnp.sqrt(0.01))

        return Graph(nodes=nodes, edges=edges, intervene_mode=False)

    def update_edges(self, graph: Graph, key: jax.Array) -> Graph:
        """"""

        w = self.edge_fn(graph, key)  # N x N x De
        #w = jnp.where(graph.edges.adj, w, 0.0)
        adj = jnp.where(w, 1.0,0.0)
        edges = graph.edges._replace(weights=w, adj=adj)
        graph = graph._replace(edges=edges)
        return graph


class RNNModel(eqx.Module):
    weights: jax.Array
    policy_iters: int
    obs_dims: int
    action_dims: int

    def __init__(self, weights: jax.Array, policy_iters: int, obs_dims: int, action_dims: int):
        self.weights = weights
        self.policy_iters = policy_iters
        self.obs_dims = obs_dims
        self.action_dims = action_dims

    def __call__(self, obs: jax.Array, rnn_state: jax.Array) -> Tuple[jax.Array, jax.Array]:
        
        #def set_input(h):
        #    h = h.at[0].set(1)
        #    h = h.at[1:self.obs_dims + 1].set(obs)
        #    return h

        def set_input(h):
            return jnp.concatenate([jnp.array([1.0]), obs, h[self.obs_dims + 1:]])

        def rnn_step(h):
            h = set_input(h)
            h = jnn.tanh(jnp.matmul(self.weights, h))
            return h

        h = jax.lax.fori_loop(0, self.policy_iters, lambda _, h: rnn_step(h), rnn_state)
        a = h[-self.action_dims:]
        return a#, h


class NDPToRNN(eqx.Module):
    ndp: GeccoNDP
    max_dev_steps: int
    action_dims: int
    obs_dims: int
    bias: bool

    def __init__(self, ndp: GeccoNDP, action_dims: int, obs_dims: int, *, max_dev_steps: int):
        self.ndp = ndp
        self.action_dims = action_dims
        self.obs_dims = obs_dims
        self.bias = False
        self.max_dev_steps = max_dev_steps

    def create_rnn(self, state: PolicyState, policy_iters) -> RNNModel:
        return RNNModel(weights=state.weights, 
                        policy_iters=policy_iters, 
                        obs_dims=self.obs_dims, 
                        action_dims=self.action_dims)

    def get_phenotype(self, key, ndp, target_function=None, eval=False, current_gen: int = 0) -> PolicyState:
        G_init = self.init_graph(key)
        G, all_graphs = ndp.rollout(G_init, key, self.max_dev_steps, target_function)

        def get_policy_state(graph):
            weights = graph.edges.weights
            return PolicyState(weights=weights,
                               rnn_state=jnp.zeros_like(graph.nodes.mask),
                               mask=graph.nodes.mask,
                               node_embedding=graph.nodes.embedding,
                               adj=graph.edges.adj)

        policy_states = jax.vmap(get_policy_state, in_axes=(0))(all_graphs)
        final_policy_state = get_policy_state(G)
        init_policy_state = get_policy_state(G_init)

        def concat_fn(x, y, z):
            return jnp.concatenate([x, y, z], axis=0)

        def expand_dims_fn(x):
            return jnp.expand_dims(x, axis=0)

        policy_states = jax.tree_map(concat_fn, 
                                     jax.tree_map(expand_dims_fn, init_policy_state), 
                                     policy_states, 
                                     jax.tree_map(expand_dims_fn, final_policy_state))

        return final_policy_state, policy_states

    def initialize(self, key: jax.Array, target_function=None, eval=False, current_gen: int = 0) -> PolicyState:
        G_init = self.init_graph(key)
        G, all_graphs = self.ndp.rollout(G_init, key, self.max_dev_steps, target_function)

        def get_policy_state(graph):
            weights = graph.edges.weights
            return PolicyState(weights=weights,
                               rnn_state=jnp.zeros_like(graph.nodes.mask),
                               mask=graph.nodes.mask,
                               node_embedding=graph.nodes.embedding,
                               adj=graph.edges.adj)

        policy_states = jax.vmap(get_policy_state, in_axes=(0))(all_graphs)
        final_policy_state = get_policy_state(G)
        init_policy_state = get_policy_state(G_init)

        def concat_fn(x, y, z):
            return jnp.concatenate([x, y, z], axis=0)

        def expand_dims_fn(x):
            return jnp.expand_dims(x, axis=0)

        policy_states = jax.tree_map(concat_fn, 
                                     jax.tree_map(expand_dims_fn, init_policy_state), 
                                     policy_states, 
                                     jax.tree_map(expand_dims_fn, final_policy_state))

        return final_policy_state, policy_states

    def init_graph(self, key: jax.Array):
        return self.ndp.initialize(key)



def make_model(config, key):
    """ Creates the NDP_newsource model.
    """

    key, key_model = jr.split(key)
    key_node, key_edge, key_div, key_edge_growth, key_edge_death = jr.split(key_model, 5)

    action_size = config["env_config"]["action_size"]
    input_size = config["env_config"]["observation_size"]
    max_nodes = config["model_config"]["model_params"]["num_hidden_neurons"] + action_size + input_size + 1

    n_features_derived = max_nodes
    n_init_nodes = action_size + input_size + 1
    n_node_features = n_features_derived

    # div_fn learns neuron mitosis
    div_fn = DivModel(key_div, n_node_features=n_node_features)
    edge_fn = EdgeModel(key_node,
                        n_node_features)

    ndp = GeccoNDP(n_init_nodes=n_init_nodes,
                   n_node_features=n_node_features,
                   edge_fn=edge_fn,
                   max_nodes=max_nodes,
                   div_fn=div_fn,
                   )  # type: ignore

    development = NDPToRNN(ndp,
                     max_dev_steps=config["model_config"]["model_params"]["dev_steps"],
                     action_dims=action_size,
                     obs_dims=input_size)

    development.initialize()

    return model
