# Machine Learning Labs

A compact collection of educational ML/DL implementations. The repository
focuses on understanding and comparing algorithms rather than wrapping them in
a production service.

## Topics

| Area | What is implemented | Directory |
| --- | --- | --- |
| Handwritten letter recognition | Feed-forward network, Hopfield associative memory, and Kohonen SOM on EMNIST Letters | [`handwritten-letter-recognition`](handwritten-letter-recognition/) |
| Reinforcement learning | DQN experiments and a tabular Q-learning baseline for MountainCar | [`mountaincar-dqn`](mountaincar-dqn/) |
| CNN optimization | Winograd F(2x2, 3x3) convolution, correctness checks, and benchmarks | [`winograd-convolution`](winograd-convolution/) |

Together, the labs cover supervised learning, associative memory,
self-organizing maps, reinforcement learning, convolution arithmetic,
evaluation, visualization, and GPU-aware PyTorch code.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

EMNIST is downloaded automatically by torchvision when a handwriting script
is run for the first time. Generated datasets, model checkpoints, caches, and
large intermediate arrays are intentionally excluded from Git.

## Scope

These are coursework-scale implementations written to explore algorithms and
their trade-offs. Each topic directory contains its own usage notes and entry
points.
