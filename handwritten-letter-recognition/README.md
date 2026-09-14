# Handwritten Letter Recognition

Experiments on the 28×28 grayscale **EMNIST Letters** dataset using several
neural-network paradigms. The scripts download the dataset through torchvision
and use CUDA when it is available.

## Labs

| Original lab | Current file | Technique |
| --- | --- | --- |
| Lab 1 | `feedforward_emnist.py` | Fully connected classifier with a configurable activation function |
| Lab 2 (simple) | `hopfield_hebbian.py` | Classical Hopfield network with Hebbian weights |
| Lab 2 | `hopfield_pseudoinverse.py` | Hopfield memory with several prototypes and pseudoinverse weights |
| Extended Lab 2 | `hopfield_robustness.py` | Hopfield retrieval and robustness under bit-flip, Gaussian, and dropout noise |
| Lab 3 | `kohonen_som.py` | Kohonen self-organizing map for clustering and classifying letters |

The Hopfield and SOM scripts default to a small subset such as `A`, `C`, `F`,
and `Z` so that the behavior remains easy to visualize.

## Examples

Train the feed-forward classifier on all 26 letters:

```bash
python3 handwritten-letter-recognition/feedforward_emnist.py
```

Evaluate selected letters with the improved Hopfield network:

```bash
python3 handwritten-letter-recognition/hopfield_pseudoinverse.py \
  --letters ACFZ --K 5 --test_samples 2000
```

Compare classical Hopfield recall with and without Gram–Schmidt
orthogonalization:

```bash
python3 handwritten-letter-recognition/hopfield_hebbian.py --letters ACFZ
python3 handwritten-letter-recognition/hopfield_hebbian.py --letters ACFZ --gs
```

Train a 12×12 Kohonen map:

```bash
python3 handwritten-letter-recognition/kohonen_som.py \
  --letters ACFZ --width 12 --height 12 --epochs 10
```

## Notes

- EMNIST labels are converted from `1..26` to zero-based class indices.
- Several scripts rotate and mirror torchvision's EMNIST images into their
  natural orientation.
- The visualizations include confusion matrices, example predictions, training
  curves, and robustness measurements.
- Model checkpoints and the downloaded `data/` directory are local artifacts
  and are excluded from Git.
