Implementation of nanoGPT from https://github.com/karpathy/ng-video-lecture in JAX.

Some alternations to ensure it works on Apple Metal. Runs on a 32GB M2 Max.

TODO:
 * Split training and model definition files
 * Implement checkpointing
 * See if there's any way to get flash attention working on Metal JAX