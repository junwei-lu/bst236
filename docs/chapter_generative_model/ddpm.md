# Diffusion Model

Understanding DDPM in Plain Language

Many articles that introduce DDPM start directly with transition distributions and variational inference, throwing a bunch of mathematical notations that scare off many readers. (Of course, from this kind of introduction we can again see that DDPM is essentially a VAE rather than a diffusion model.) On top of the traditional impression of diffusion models being difficult, this creates the illusion that DDPM requires very advanced mathematical knowledge.

In fact, DDPM can be understood in a very plain and intuitive way. It’s not harder than GANs, which are often explained with the intuitive analogy of “forger vs. discriminator.”

From Noise to Data

Let’s say we want to build a generative model like a GAN. It’s essentially a process that transforms a random noise vector $z$ into a data sample $x$:

Random noise z ≈ bricks and cement
          ↓
      Transformation
          ↓
  Sample data x ≈ Skyscraper

Call Me an Engineer

We can imagine this process as “construction,” where the random noise $z$ is raw materials like bricks and cement, and the data sample $x$ is the skyscraper. So, a generative model is like a construction crew that builds skyscrapers from raw materials.

This process is hard, which is why there’s so much research on generative models. But as the saying goes, “destruction is easier than construction.” Maybe you can’t build a skyscraper, but you can definitely tear one down. So let’s think about the reverse process of dismantling a skyscraper into bricks and cement.

Let $x_0$ be the finished skyscraper (data sample), and $x_T$ be the pile of bricks and cement (random noise). Assume it takes $T$ steps to dismantle it. The entire process is:

$$
x = x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \cdots \rightarrow x_{T-1} \rightarrow x_T = z
$$

The challenge of building a skyscraper is that going from raw materials $x_T$ to the final structure $x_0$ is too big a leap. But if we have the intermediate states $x_1, x_2, \dots, x_T$, we can understand how to go from one step to the next.

So, if we know the transformation $x_{t-1} \rightarrow x_t$ (dismantling), then reversing it $x_t \rightarrow x_{t-1}$ is like constructing. If we can learn the reverse function $\mu(x_t)$, then starting from $x_T$, we can repeatedly apply $\mu$ to reconstruct the skyscraper:

$$
x_{T-1} = \mu(x_T),\quad x_{T-2} = \mu(x_{T-1}), \dots
$$

How to Dismantle

As the saying goes, “one bite at a time.” DDPM follows this principle by defining a gradual transformation from data samples to noise (dismantling), then learns the reverse (construction). So it’s more accurate to call DDPM a “gradual model” rather than a “diffusion model.”

Specifically, DDPM defines the dismantling process as:

$$
x_t = \alpha_t x_{t-1} + \beta_t \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, I)
$$

Here, $\alpha_t, \beta_t > 0$ and $\alpha_t^2 + \beta_t^2 = 1$. Typically, $\beta_t$ is close to 0, representing small degradation at each step. The noise $\varepsilon_t$ adds randomness—think of it as raw material injected at each step.

Repeating this step, we get:

$$
x_t = (\alpha_t \cdots \alpha_1)x_0 + \text{a weighted sum of Gaussian noises}
$$

Why do we require $\alpha_t^2 + \beta_t^2 = 1$? Because the Gaussian noise sum then becomes a single Gaussian with mean 0 and total variance 1, i.e.,

$$
x_t = \bar{\alpha}_t x_0 + \sqrt{1 - \bar{\alpha}_t^2} \bar{\varepsilon}_t, \quad \bar{\varepsilon}_t \sim \mathcal{N}(0, I)
$$

This makes computing $x_t$ very convenient. Furthermore, $\bar{\alpha}_T \approx 0$, meaning that after $T$ steps, only noise remains.

How to Construct

Now that we have data pairs $(x_{t-1}, x_t)$ from dismantling, we can learn the reverse $x_t \rightarrow x_{t-1}$ via a model $\mu(x_t)$. The loss is:

$$
|x_{t-1} - \mu(x_t)|^2
$$

From the dismantling equation:

$$
x_{t-1} = \frac{1}{\alpha_t}(x_t - \beta_t \varepsilon_t)
$$

We design:

$$
\mu(x_t) = \frac{1}{\alpha_t}(x_t - \beta_t \varepsilon_\theta(x_t, t))
$$

The loss becomes:

$$
\frac{\beta_t^2}{\alpha_t^2} |\varepsilon_t - \varepsilon_\theta(x_t, t)|^2
$$

Using the earlier expression for ****$x_t$:

$$
x_t = \bar{\alpha}_t x_0 + \text{weighted noises}
$$

We get the training loss:

$$
|\varepsilon_t - \varepsilon_\theta(\bar{\alpha}_t x_0 + \text{noises}, t)|^2
$$

But this involves too many random variables: $x_0, \varepsilon_t, \varepsilon_{t-1}, t$. Too many random variables increase variance. Luckily, using Gaussian addition tricks, we can combine $\varepsilon_t$ and $\varepsilon_{t-1}$ into one $\varepsilon$:

$$
\varepsilon_t = \frac{\beta_t \varepsilon - \alpha_t \bar{\beta}_{t-1} \omega}{\bar{\beta}_t}, \quad \omega \sim \mathcal{N}(0, I)
$$

Substitute back and simplify to get the final DDPM loss:

$$
\left|\varepsilon - \frac{\bar{\beta}t}{\beta_t} \varepsilon\theta(\bar{\alpha}_t x_0 + \bar{\beta}_t \varepsilon, t)\right|^2
$$

(Note: In the original paper, their $\varepsilon_\theta$ absorbs the $\bar{\beta}_t/\beta_t$ term.)

Recursive Generation

Once trained, DDPM generates samples by starting from $x_T \sim \mathcal{N}(0, I)$ and running:

$$
x_{t-1} = \frac{1}{\alpha_t}(x_t - \beta_t \varepsilon_\theta(x_t, t))
$$

This corresponds to greedy decoding. For random sampling, we add noise:

$$
x_{t-1} = \frac{1}{\alpha_t}(x_t - \beta_t \varepsilon_\theta(x_t, t)) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
$$

Usually, $\sigma_t = \beta_t$ to keep variance consistent.

DDPM vs PixelRNN/PixelCNN

DDPM needs $T=1000$ steps to generate one image, making it slow. Many works aim to speed it up. Readers may compare this with PixelRNN/PixelCNN, which also generate images slowly in an autoregressive manner.

The key difference: PixelRNN/PixelCNN generate pixel by pixel in a predefined order. This introduces strong inductive bias, and the result depends heavily on the pixel ordering. In contrast, DDPM’s generation is symmetric and order-free—every pixel is treated equally, reducing inductive bias and improving results.

Also, PixelRNN/PixelCNN’s generation steps scale with resolution (width × height × channels), while DDPM uses a fixed number of steps $T$. So DDPM is faster for high-resolution images.

In summary, DDPM is not as complicated as it seems. Using the dismantle-reconstruct analogy and basic probability, we derived everything without needing variational inference or energy-based models. It’s an intuitive yet powerful approach to generative modeling.