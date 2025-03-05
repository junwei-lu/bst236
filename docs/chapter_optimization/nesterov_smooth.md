# Nesterov's Smoothing

We introduced the proximal algorithm to solve problems of the form:

$$
\min_x f(x) + h(x)
$$

When $f$ is smooth, the accelerated proximal gradient descent has a convergence rate of $O(1/t^2)$. What if the objective is not smooth? For example, the square-root Lasso:

$$
\min_\beta \|Y - \mathbb X\beta\|_2 + \lambda\|\beta\|_1
$$

The $\ell_2$-norm $\|x\|_2$ is not differentiable at $0$. Nesterov's smoothing idea involves approximating the non-smooth objective function with a smooth one and minimizing the smooth approximation using gradient descent.

**Definition**: A convex function $f$ is $(\alpha, \beta)$-smoothable if for any $\mu > 0$, there exists a convex approximation $f_{\mu}$ such that:

1. $f_{\mu}(x) \leq f(x) \leq f_{\mu}(x) + \beta\mu$, for all $x$.
2. $f_{\mu}$ is $\frac{\alpha}{\mu}$-smooth.

### Example: $\ell_1$-norm

Approximate the absolute value $f(z) = |z|$ using the Huber loss:

$$
h_{\mu}(z) = \begin{cases} z^2/(2\mu), &\text{if } |z|\leq \mu; \\ |z| - \mu/2, &\text{otherwise}. \end{cases}
$$

The Huber loss is $\frac{1}{\mu}$-smooth, making $|z|$ $(1, \frac{1}{2})$-smoothable. For the $\ell_1$-norm $f(z) = \|z\|_1$, we approximate it by $\sum_{i=1}^d h_{\mu}(z_i)$, making it $(1, \frac{d}{2})$-smoothable.

### Example: $\ell_2$-norm

Approximate the $\ell_2$-norm $f(x) = \|x\|_2$ by:

$$
f_{\mu}(x) = \sqrt{\|x\|_2^2 + \mu^2} - \mu
$$

This makes $\|x\|_2$ $(1,1)$-smoothable, with dimension-free smoothing parameters.

The following theorem shows the convergence rate using Nesterov's smoothing idea:

**Theorem**: Given $F(x) = f(x) + h(x)$, where $f$ is $(\alpha, \beta)$-smoothable and $h$ is convex, let $f_\mu$ be the $\frac{1}{\mu}$-smooth approximation to $f$. Applying accelerated proximal gradient descent to $F_\mu(x) = f_{\mu}(x) + h(x)$ with $\mu = \epsilon/(2\beta)$ achieves $\epsilon$-accuracy if $t \gtrsim \frac{\sqrt{\alpha\beta}}{\epsilon}$.

For the square-root Lasso, applying accelerated proximal gradient descent to:

$$
\min_{\beta} \sqrt{\|Y  - \mathbb X \beta\|_2^2 + \mu^2} + \lambda\|\beta\|_1
$$

achieves $\epsilon$-accuracy within $O(1/\epsilon)$ steps, assuming the maximum singular value of the design matrix $\mathbb{X}$ is bounded. 