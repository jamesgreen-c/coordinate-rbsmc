import jax

def block_until_ready_tree(x):
    return jax.tree_util.tree_map(
        lambda y: y.block_until_ready() if hasattr(y, "block_until_ready") else y,
        x,
    )