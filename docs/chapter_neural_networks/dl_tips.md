# Tips for Training Neural Networks

The following are tips for training neural networks based on Andrej Karpathy's [blog post](https://karpathy.github.io/2019/04/25/recipe/).

## General Philosophy

- **Neural networks are a leaky abstraction**:

      - Despite simple APIs suggesting plug-and-play convenience, successful training requires deep understanding
      - Common libraries hide complexity but don't eliminate the need for foundational knowledge

- **Neural networks fail silently**:

      - Errors are typically logical rather than syntactical, rarely triggering immediate exceptions
      - Mistakes often subtly degrade performance, making debugging challenging

## Recommended Training Process

### Core Principles

- Patience and attention to detail are critical
- Start from simplest possible setup, progressively adding complexity
- Continuously validate assumptions and visualize intermediate results

## Deep Learning Recipe

### Step 1: Data Exploration

- Spend substantial time manually inspecting data
- Look for the following data issues:
  
      - Duplicate or corrupted data
      - Imbalanced distributions
      - Biases and potential patterns

- Think qualitatively:

      - Which features seem important?
      - Can you remove noise or irrelevant information?
      - Understand label quality and noise level

- Write simple scripts to filter/sort and visualize data distributions/outliers

### Step 2: Build Training Code

- Build a basic training/evaluation pipeline using a simple model (linear classifier or small CNN)
- Best practices at this stage:

      - Fix random seed: Ensures reproducibility
      - Simplify: Disable data augmentation or other complexity initially
      - Full test evaluation: Run tests on entire datasets; avoid reliance on smoothing metrics
      - Verify loss at initialization: Ensure correct loss values at initialization (e.g., -log(1/n_classes))
      - Proper Initialization: Set final layer biases according to data distribution
      - Human baseline: Measure human-level accuracy for reference
      - Input-independent baseline: Check that your model learns something beyond trivial solutions
      - Overfit a tiny dataset: Confirm your network can achieve near-zero loss on a single batch—essential debugging step
      - Visualizations: Regularly visualize inputs immediately before entering the model, predictions over training time, and loss dynamics
      - Check dependencies with gradients: Use backward pass to debug vectorization and broadcasting issues
      - Generalize incrementally: Write simple, explicit implementations first; vectorize/generalize later

### Step 3: Overfitting 

- Goal of the overfitting first for smaller batch size is to:

      - Ensure your model can at least perfectly memorize the training set (overfit)
      - If your model can't overfit, you likely have bugs or incorrect assumptions
- Tips for this stage:

      - Don't be creative initially: Use established architectures first (e.g., ResNet-50 for images)
      - Use Adam optimizer initially: Learning rate ~ 3e-4; Adam is forgiving compared to SGD
      - Add complexity carefully: Introduce inputs or complexity gradually
      - Disable learning rate decay initially: Use a fixed learning rate until very late

### Step 4: Regularization 

- Once you've achieved overfitting, regularize to improve validation accuracy
- Recommended regularization methods:

      - Add more real data: Most effective regularizer
      - Data augmentation: Simple augmentation can dramatically help
      - Creative augmentation: Domain randomization, synthetic/partially synthetic data, GAN-generated samples
      - Pretrained models: Almost always beneficial
      - Avoid overly ambitious unsupervised methods: Stick with supervised learning
      - Reduce input dimensionality: Eliminate irrelevant or noisy features
      - Simplify architecture: Remove unnecessary parameters (e.g., average pooling vs. fully connected layers)
      - Smaller batch size: Stronger regularization effect with batch normalization
      - Dropout carefully: Use spatial dropout (dropout2d) for CNNs cautiously (interacts badly with batch norm)
      - Weight decay: Strengthen weight regularization
      - Early stopping: Halt training based on validation performance
      - Larger model with early stopping: Larger networks sometimes generalize better when stopped early

### Step 5: Hyperparameter Tuning

- Use random search, not grid search: Better coverage, especially with many hyperparameters
- Bayesian optimization tools can help, but practical gains might be limited. Focus on intuition and experimentation

Tune one by one ranking from most important to least important:

- Learning rate
- Weight decay
- Dropout rate
- Batch size

### Step 6: Squeezing Out Final Performance

- Ensemble methods: Reliable 2% accuracy improvement
- Distillation: Compress ensemble into single model (knowledge distillation)
- Longer training: Don't prematurely stop training; often performance continues improving subtly

### Final Checks

- Visualize first-layer weights (should look meaningful, not random noise)
- Check intermediate activations for artifacts or unexpected patterns

## Conclusion 

By following this disciplined process, you will:
- Deeply understand your data and problem
- Have high confidence in the correctness of your pipeline
- Systematically explore complexity, building intuition and trust at each step

This methodical approach drastically reduces debugging complexity and increases your likelihood of achieving state-of-the-art results.