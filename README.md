# What is Synthetic Data?
Synthetic data is artificially generated data that is not collected from real world events. It replicates the statistical components of real data without containing any identifiable information, ensuring individuals' privacy.

# What is a Genetic Algorithm?
Genetic Algorithms are algorithms that mimic natural selection and evolution and it is widely applied for optimization problems. The most basic version starts with a randomly generated population of candidate solutions (chromosomes), and through the use of genetic operators such as selection, crossover, and mutation, it evolves these individuals towards better solutions over time, based on a fitness function that measures their quality. 

### Structure 
```bash
.
├── src
│   └── ydata_synthetic
│       ├── genetic_algorithm
│       ├── preprocessing
│       └── synthesizers
└── examples
```

### In this repo you can find the following GAN architectures:
- [CGAN (Conditional GAN)](https://arxiv.org/abs/1411.1784)
- [WGAN-GP (Wassertein GAN with Gradient Penalty)](https://arxiv.org/abs/1704.00028)
