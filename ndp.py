from typing import Callable, Union, Optional
from typing import Optional, Tuple, NamedTuple
from jaxtyping import Float, Array, PyTree, Int, Bool
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import PyTree

State = PyTree[...]
Params = PyTree[...]
Statics = PyTree[...]


class Node(NamedTuple):
    # -------------------------------------------------------------------
    mask: Optional[Float[Array, "N"]] = None
    mask_apoptosis: Optional[Float[Array, "N"]] = None
    cell_grow_interv: Optional[Float[Array, "N"]] = None
    edge_grow_interv: Optional[Float[Array, "N"]] = None
    edge_death_interv: Optional[Float[Array, "N"]] = None

    diffed: Optional[Float[Array, "N"]] = None
    embedding: Optional[Float[Array, "N"]] = None
    node_type: Optional[Float[Array, "N"]] = None
    p: Optional[Float[Array, "N X"]] = None
    pholder: Optional[PyTree] = None



#-------------------------------------------------------------------
class Edge(NamedTuple):
    #-------------------------------------------------------------------
    adj: Optional[Float[Array, "..."]]=None
    weights: Optional[Float[Array, "E De"]]=None
    spatial_grid: Optional[Float[Array, "..."]] = None


class Graph(NamedTuple):
	#-------------------------------------------------------------------
    intervene_mode: bool
    nodes: Node
    edges: Edge
    pholder: Optional[PyTree]=None
#-------------------------------------------------------------------


class BaseModel(eqx.Module):
    """
    """

    def dna_partition(self):
        raise NotImplementedError("This model has no dna partition")

    # -------------------------------------------------------------------

    def partition(self):
        return eqx.partition(self, eqx.is_array)

    # -------------------------------------------------------------------

    def save(self, filename: str):
        eqx.tree_serialise_leaves(filename, self)

    # -------------------------------------------------------------------

    def load(self, filename: str):
        return eqx.tree_deserialise_leaves(filename, self)


class DevelopmentalModel(BaseModel):
    """
    Base structure for iterative developmental models
    """

    # -------------------------------------------------------------------
    # -------------------------------------------------------------------

    def __init__(self):
        pass

    # -------------------------------------------------------------------

    def __call__(self, state: State, key: jr.PRNGKey) -> State:
        raise NotImplementedError("__call__ method not implemented")

    # -------------------------------------------------------------------

    def initialize(self, key: jr.PRNGKey) -> State:
        raise NotImplementedError("initialize method not implemented")

    # -------------------------------------------------------------------

    def init_and_rollout(self, key: jr.PRNGKey, steps: int) -> State:
        key_init, key_rollout = jr.split(key)
        state = self.initialize(key_init)
        return self.rollout(state, key_rollout, steps)

    # -------------------------------------------------------------------


    def rollout(self, state: State, key: jr.PRNGKey, steps: int, key_fixed, target_function) -> Tuple[State, State]:
        pass

    # -------------------------------------------------------------------

    def init_and_rollout_(self, key: jr.PRNGKey, steps: int) -> State:
        key_init, key_rollout = jr.split(key)
        state = self.initialize(key_init)
        return self.rollout_(state, key_rollout, steps)

    # -------------------------------------------------------------------

    def rollout_(self, state: State, key: jr.PRNGKey, steps: int) -> State:
        def _step(i, sk):
            s, k = sk
            k, k_ = jr.split(k)
            s = self.__call__(s, k_)
            return [s, k]

        [state, _] = jax.lax.fori_loop(0, steps, _step, [state, key])
        return state


class BaseNDP(DevelopmentalModel):
    """
    """
    n_node_features: int
    max_nodes : int
    n_init_nodes: int
    node_fn: Union[PyTree, Callable]
    edge_fn: Union[PyTree, Callable]
    div_fn: Union[PyTree, Callable]    # Statics:
    # -------------------------------------------------------------------


    # -------------------------------------------------------------------

    def __init__(self, node_fn, edge_fn, div_fn,
                 n_node_features, max_nodes, n_init_nodes
                 ):
        self.edge_fn = edge_fn
        self.node_fn = node_fn
        self.div_fn = div_fn
        self.n_node_features =n_node_features
        self.max_nodes = max_nodes
        self.n_init_nodes = n_init_nodes



class NDP(BaseNDP):
    """
    """
    grid_size : int
    use_location: bool
    use_apoptosis: bool
    use_diff: bool
    diff_fn: Union[PyTree, Callable]
    decide_type_fn: Union[PyTree, Callable]
    synapse_death_fn: Union[PyTree, Callable]
    ''


    def __init__(self, edge_fn, div_fn, use_diff, diff_fn,use_location, use_apoptosis, grid_size, n_node_features, synapse_growth_fn,synapse_death_fn, decide_type_fn, max_nodes
                 ):
        self.grid_size = grid_size
        self.use_location = use_location
        self.use_apoptosis = use_apoptosis
        self.diff_fn = diff_fn
        self.use_diff = use_diff
        self.decide_type_fn = decide_type_fn
        self.synapse_death_fn=synapse_death_fn

        super().__init__(edge_fn=edge_fn,
                       node_fn=None,
                       div_fn=div_fn,
                       max_nodes=max_nodes,
                       decide_type_fn=decide_type_fn,
                       n_node_features=n_node_features,
                       synapse_growth_fn=synapse_growth_fn)


    # -------------------------------------------------------------------

    def __call__(self, graph: Graph, key: jax.Array, counter: int, dev_step, index, current_key_fixed, target_function) -> Graph:
        graph = self.add_new_nodes(graph, current_key_fixed, counter=counter, dev_step=dev_step, index=index, target_function=target_function)
        graph = self.update_edges(graph, key)

        return graph


    def update_locations(self, key, mask, grow, adj, spatial_grid, locs):
        N = mask.shape[0]
        indexes = jnp.arange(N)

        n_alive = mask.sum()
        n_grow = grow.sum()
        new_n = n_alive + n_grow
        new_mask = (jnp.arange(adj.shape[0]) < new_n).astype(float)
        xnew_mask = new_mask - mask

        n_notgrow = jnp.sum(mask) - grow.sum()
        parents = jnp.ones([N, ]) * (-100)  # index i tells you who is the kid of cell i

        def set_parent(i):
            temp = parents.at[i + n_notgrow.astype(jnp.int32)].set(i + n_alive)
            return temp[i + n_notgrow.astype(jnp.int32)]

        parents = jax.vmap(set_parent)(indexes)

        parents = parents.astype(jnp.int32)

        parents = jnp.where(mask, parents, 0)

        def trace_path(loc, angle):
            x0, y0 = loc

            epsilon = 0.0001
            radian = jnp.radians(angle)
            dx = jnp.cos(radian) + epsilon
            dy = jnp.sin(radian) + epsilon

            def move_to_next_cell(x, y, dx, dy):
                # Calculate t_max_x
                t_max_x = jax.lax.cond(
                    dx > 0,
                    lambda _: (jnp.floor(x) + 1 - x) / dx,
                    lambda _: (jnp.ceil(x) - 1 - x) / dx,
                    operand=None
                )

                # Calculate t_max_y
                t_max_y = jax.lax.cond(
                    dy > 0,
                    lambda _: (jnp.floor(y) + 1 - y) / dy,
                    lambda _: (jnp.ceil(y) - 1 - y) / dy,
                    operand=None
                )

                # Update x and y based on which t_max is smaller
                x, y = jax.lax.cond(
                    t_max_x < t_max_y,
                    lambda _: (x + dx * t_max_x, y + dy * t_max_x),
                    lambda _: (x + dx * t_max_y, y + dy * t_max_y),
                    operand=None
                )

                return x, y

            # Function to check if x, y are within bounds
            def in_bounds(x, y, grid_size):
                # Use JAX's logical_and to check bounds
                return jnp.logical_and(0 < x, x <= grid_size) & jnp.logical_and(0 < y, y <= grid_size)

            # Initial state: x, y, and an empty array to track cells crossed
            def cond_fn(state):
                x, y, cells_crossed = state
                return in_bounds(x, y, self.grid_size)

            def body_fn(state):
                x, y, cells_crossed = state
                x_next, y_next = move_to_next_cell(x, y, dx, dy)

                # Check if the new position is within bounds
                is_in_bounds = in_bounds(x_next, y_next, self.grid_size)

                # Convert coordinates to integers
                int_x, int_y = x_next, y_next

                # Update the cells crossed if new position is within bounds
                cell_tuple = jnp.array([int_x, int_y])
                # temp = jnp.vstack([cells_crossed, jnp.array([int_x, int_y])])

                nonzero_rows = jnp.any(cells_crossed != 0, axis=1)
                n_alive = jnp.sum(nonzero_rows)

                not_already_exists = jnp.logical_not(jnp.any(jnp.all(cells_crossed == cell_tuple, axis=1)))
                # is_in_bounds & already_exists
                cells_crossed = jax.lax.cond(
                    is_in_bounds & not_already_exists,
                    lambda cells: cells.at[n_alive].set(cell_tuple),
                    lambda cells: cells,
                    cells_crossed
                )

                return x_next, y_next, cells_crossed

            # Initial state
            initial_state = (x0, y0, jnp.zeros((self.grid_size, 2), dtype=jnp.int32))

            # Run the while loop
            final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)

            return final_state[2]  # Return the list of cells crossed

        def find_empty_cell(coordinates):
            # Apply the vectorized check to all coordinates
            checks = vectorized_check(coordinates)

            # Find the index of the first nonzero check
            first_nonzero_index = jnp.argmax(checks)  # `argmax` returns the index of the first True value

            # Retrieve the first coordinate with a nonzero value
            first_nonzero_coordinate = coordinates[first_nonzero_index]

            return first_nonzero_coordinate

        directions = jax.random.randint(key, shape=((N,)), minval=0, maxval=360)

        crossed_cells = jax.vmap(trace_path)(locs, directions)  # which cells on the grid does this neuron cross?

        # Function to check if a coordinate corresponds to a nonzero value in the grid
        def is_nonzero_at_coordinate(coord):
            row, col = coord
            return spatial_grid[row, col] != 1

        # Vectorize the check function to apply it to each coordinate
        vectorized_check = jax.vmap(is_nonzero_at_coordinate)

        # which ones are free
        updated_locs = jax.vmap(find_empty_cell)(crossed_cells)

        def set_loc(updated_loc, parent):
            updated = locs.at[parent].set(updated_loc)
            return updated

        new_locs = jax.vmap(set_loc, in_axes=(0, 0))(updated_locs, parents).astype(jnp.int32)

        def get_new_loc(loc, parent):
            return loc[parent, ...]

        new_locs = jax.vmap(get_new_loc)(new_locs, parents)

        # Reorder the rows of `arr` based on the `order` array
        tnew_locs = jnp.empty_like(new_locs)

        # Place rows of `arr` in the positions specified by `order`
        tnew_locs = tnew_locs.at[parents].set(new_locs)

        # tnew_locs = new_locs[parents]
        new_locs = jnp.where(xnew_mask[:, None], tnew_locs, locs).astype(jnp.int32)

        def update_grid(loc):
            # new_spatial_grid = jnp.where(loc, 1, spatial_grid)
            temp_new_spatial_grid = spatial_grid.at[loc[0], loc[1]].set(1)

            new_spatial_grid = jnp.where(jnp.sum(loc), temp_new_spatial_grid, spatial_grid)

            # new_spatial_grid = jax.lax.cond(jnp.any(loc), lambda x: x.at[loc].set(1), lambda x: x, spatial_grid )
            return new_spatial_grid

        new_grids = jax.vmap(update_grid)(new_locs)
        spatial_grid = jnp.sum(new_grids, axis=0)
        spatial_grid = jnp.where(spatial_grid, 1, 0)

        return new_locs, spatial_grid


    def add_new_nodes(self, graph: Graph, key: jax.Array, counter: int, dev_step: int, index:int, target_function: int) -> Graph:
        """"""

        # each cell decides whether to grow another cell
        grow = jnp.squeeze(self.div_fn(graph, key))

        alive = graph.nodes.mask*graph.nodes.mask_apoptosis
        grow = grow * alive*(1-graph.nodes.diffed)

        n_grow = grow.sum()
        N = grow.shape[0]
        n_alive = alive.sum()
        new_n = n_alive + n_grow
        new_mask = (jnp.arange(graph.edges.adj.shape[0]) < new_n).astype(float)
        xnew_mask = new_mask - graph.nodes.mask
        new_mask = jnp.where(graph.nodes.mask_apoptosis, new_mask, 0.0)

        # compute childs index for each parent: pc[parent id] = child id
        pc = (jnp.where(grow, jnp.cumsum(grow) - 1, -1) + (n_alive * grow)).astype(int)

        # Set child´s incoming connections (parent neighbors + parents)
        nA = jax.ops.segment_sum(jnp.identity(N), pc, N).T

        adj = jnp.where(xnew_mask[None, :], nA, graph.edges.adj) * new_mask[None, :] * new_mask[:, None]

        # each synapse decides whether to form
        synapse_growth = self.synapse_growth_fn(graph, key)
        mask_for_edges = new_mask*graph.nodes.mask_apoptosis
        mask_edges_row = jnp.expand_dims(mask_for_edges, axis=1).repeat(N,axis=1)
        mask_edges_col= jnp.expand_dims(mask_for_edges,axis=0)
        mask_edges_col = jnp.repeat(mask_edges_col, repeats=N,axis=0)
        mask_edges = jnp.logical_and(mask_edges_col, mask_edges_row)

        adj = jnp.where(synapse_growth, 1, adj)
        adj = jnp.where(mask_edges, adj, 0)

        # synapse death
        synapse_death = self.synapse_death_fn(graph, key)
        adj = jnp.where(synapse_death, 0, adj)

        # differentiation
        diffed = jnp.squeeze(self.diff_fn(graph, key))
        diffed = jnp.where(graph.nodes.diffed, graph.nodes.diffed, diffed) # a differentiated node cannot go back

        # cells without connections die forever
        sum_row = jnp.sum(adj, axis=0)
        sum_col = jnp.sum(adj, axis=1)
        alive = sum_row + sum_col
        no_connections = jnp.where(alive, 0.0, 1.0)

        mask_apoptosis = jnp.where(jnp.logical_and(counter > 5, no_connections), 0.0,
                                   graph.nodes.mask_apoptosis)
        new_mask = jnp.where(mask_apoptosis, new_mask, 0)



        # update node embeddings
        new_embeddings, spatial_grid = self.update_embeddings(graph.nodes.embedding, adj, grow, pc, graph.nodes.mask, graph.edges.spatial_grid, key)

        nodes = graph.nodes._replace( diffed=diffed,mask=new_mask.astype(float), mask_apoptosis=mask_apoptosis,embedding=new_embeddings)

        return graph._replace(nodes=nodes, edges=graph.edges._replace(adj=adj, spatial_grid=spatial_grid))


    def rollout(self, state: State, key: jr.PRNGKey, steps: int, max_steps, key_fixed, target_function) -> Tuple[State, State]:
        def _step(c, x):
            s, k, key_fixed, counter, dev_step, index = c
            key_fixed, current_key_fixed  =jax.random.split(key_fixed)
            counter += 1
            k, k_ = jr.split(k)

            #"""
            s =  jax.lax.cond(counter < steps,
                         lambda x : self.__call__(x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
                         lambda x: x[0],
                         (s, k_, counter, dev_step, index, current_key_fixed, target_function)
                         )
            #
            #s = self.__call__(s, k_, counter, dev_step, index, current_key_fixed, target_function)
            index += dev_step
            dev_step *= 2
            return [s, k, key_fixed, counter, dev_step, index], s


        [state, _, _,counter,_,_], states = jax.lax.scan(
            _step, [state, key, key_fixed, 0,1,0 ], None, max_steps
        )

        #state = self.update_edges(state, key)

        state = self.decide_type(state, key)


        return state, states


    def decide_type(self, graph: Graph, key: jax.Array) -> Graph:
        types = self.decide_type_fn(graph, key)
        new_nodes = graph.nodes._replace(node_type=types)
        return graph._replace(nodes=new_nodes)

    def init_embeddings(self, adj, key):
        """ Derived embeddings are the non-learned ones. It includes intrinsic and structural features
        """
        # intrinsic embedding is a random value that is inherited about division
        sum_row = jnp.sum(adj, axis=0)
        sum_col = jnp.sum(adj, axis=1)
        N = adj.shape[0]
        intrinsic_embedding = jnp.zeros((N,))
        intrinsic_embedding = intrinsic_embedding.at[0].set(jax.random.uniform(key,()))

        # age
        age = jnp.zeros((self.max_nodes,))

        # graph-features are in-degree and out-degree
        graph_features = jnp.stack((sum_row, sum_col), axis=1)

        if self.use_location:

            # spatial loc
            center = int(self.grid_size / 2)
            init_locs = jnp.zeros((self.max_nodes, 2)).at[0].set([center, center])
            embeddings = jnp.concatenate(
                (intrinsic_embedding[:, None], age[:, None], graph_features, init_locs), axis=1)

        else:
            embeddings = jnp.concatenate(
                (intrinsic_embedding[:, None], age[:, None], graph_features), axis=1)

        return embeddings

    def update_embeddings(self, embeddings, adj, grow, pc, mask, spatial_grid, key):
        # inherit intrinsic embedding from parent
        sum_row = jnp.sum(adj, axis=0)
        sum_col = jnp.sum(adj, axis=1)

        children = jnp.where(grow, pc, -100)
        parent_intrinsic = embeddings[..., 0]
        noise = jax.random.normal(key, parent_intrinsic.shape)*0.1
        embeddings = embeddings.at[children,0].set(parent_intrinsic + noise)

        # update age
        prev_age = embeddings[:, 1]
        age = prev_age + 1
        embeddings = embeddings.at[:, 1].set(age)

        # update graph features
        graph_features = jnp.stack((sum_row, sum_col), axis=1)
        embeddings = embeddings.at[:, 2:4].set(graph_features)

        # update spatial locations
        if self.use_location:
            locs = embeddings[:, -2:]
            new_locs, spatial_grid = self.update_locations(key, mask, grow, adj, spatial_grid, locs)

            embeddings = embeddings.at[:,4:].set(new_locs)

        return embeddings, spatial_grid


    def initialize(self, key):

        init_adj = jnp.zeros((self.max_nodes, self.max_nodes))
        init_emb = self.init_embeddings(init_adj, key)

        # start with a single neuron at the center of the grid
        center = int(self.grid_size / 2)
        init_grid = jnp.zeros((self.grid_size, self.grid_size)).at[center, center].set(1).astype(jnp.int32)

        nodes = Node(
            node_type=jnp.zeros((self.max_nodes,)),
            embedding=init_emb,
            diffed=jnp.zeros((self.max_nodes,)),
            cell_grow_interv=jnp.zeros((self.max_nodes,)),
            edge_grow_interv=jnp.zeros((self.max_nodes,)),
            edge_death_interv=jnp.zeros((self.max_nodes,)),
            mask=jnp.zeros((self.max_nodes,)).at[0].set(1.),
            mask_apoptosis=jnp.ones((self.max_nodes,)))

        edges = Edge(
            adj=init_adj,
            spatial_grid=init_grid,
            weights=jr.normal(key, (self.max_nodes, self.max_nodes)) * jnp.sqrt(0.01))


        return Graph(nodes=nodes, edges=edges, intervene_mode=False)


    def update_edges(self, graph: Graph, key: jax.Array) -> Graph:
        """"""

        w = self.edge_fn( graph, key)  # N x N x De

        w = jnp.where(graph.edges.adj, w, 0.0)

        n = w.shape[0]
        pre = jnp.repeat(graph.nodes.diffed[:, None], n, axis=1)
        post = jnp.repeat(graph.nodes.diffed[None, :], n, axis=0)
        diffed = jnp.logical_and(pre,post)

        w = jnp.where(diffed, w, 0.0)

        edges = graph.edges._replace(weights=w)

        graph = graph._replace(edges=edges)

        return graph
