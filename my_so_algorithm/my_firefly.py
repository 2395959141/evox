import jax
import jax.numpy as jnp
from evox import Algorithm, State, jit_class

@jit_class
class FireflyAlgorithm(Algorithm):
    def __init__(self, lb, ub, pop_size, beta_0, gamma, alpha):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.beta_0 = beta_0
        self.gamma = gamma
        self.alpha = alpha

    def setup(self, key):
        key, subkey = jax.random.split(key)
        population = jax.random.uniform(subkey, shape=(self.pop_size, self.dim), minval=self.lb, maxval=self.ub)
        brightness = jnp.zeros(self.pop_size)  # 亮度初始化为0
        return State(
            population=population,
            brightness=brightness,
            key=key,
        )

    def ask(self, state):
        return state.population, state

    def update_position(self, firefly_i, firefly_j, brightness_i, brightness_j):
        distance = jnp.linalg.norm(firefly_i - firefly_j)
        beta = self.beta_0 * jnp.exp(-self.gamma * distance**2)
        new_position = firefly_i + beta * (firefly_j - firefly_i) + self.alpha * jax.random.normal(jax.random.PRNGKey(0), shape=firefly_i.shape)
        return jnp.clip(new_position, self.lb, self.ub)

    def update_population(self, population, brightness):
        # 遍历所有萤火虫，进行位置更新
        def update_for_one(firefly_i, brightness_i):
            def update_for_j(firefly_j, brightness_j):
                return self.update_position(firefly_i, firefly_j, brightness_i, brightness_j)

            new_positions = jax.vmap(update_for_j, in_axes=(0, 0))(population, brightness)
            return jnp.mean(new_positions, axis=0)  # 求取新位置的平均值

        # 使用 vmap 对所有萤火虫进行位置更新
        new_population = jax.vmap(update_for_one, in_axes=(0, 0))(population, brightness)
        return new_population

    def tell(self, state, fitness):
        # 更新亮度
        brightness = 1 / (1 + fitness)  # 假设亮度是适应度的函数

        # 更新萤火虫的位置
        new_population = self.update_population(state.population, brightness)

        return state.replace(population=new_population, brightness=brightness)
