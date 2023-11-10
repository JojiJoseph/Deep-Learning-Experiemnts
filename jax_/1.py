import jax
import jax.numpy as jnp

# create a vector
v = jnp.array([1., 2, 3])
print(v)
print(v.__repr__())

# Finding out gradient of sum of squares
def sum_of_squares(x):
    return jnp.sum(x**2)
print("sum of squares: ", sum_of_squares(v))

# Finding out gradient of sum of squares
gradients = jax.grad(sum_of_squares)(v)
print("gradients: ", gradients)

print("\nCalculating in another way:\n")

@jax.value_and_grad
def sum_of_squares(x):
    return jnp.sum(x**2)

val, grad = sum_of_squares(v)
print(val, grad)
