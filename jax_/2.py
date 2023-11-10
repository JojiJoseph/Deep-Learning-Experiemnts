import jax
import jax.numpy as jnp

def f(x):
    return 2*x
print(jax.make_jaxpr(f)(30.0))