# RbmSAT solver

The reproduction of the RbmSAT solver for MaxSAT using Pytorch, from [https://arxiv.org/abs/2311.02101](https://arxiv.org/abs/2311.02101).

[Data Example](https://drive.google.com/file/d/1OchnxDoQ--F4aVxrQgvM-o0nSW19gnDy/view)

Usage: 

```bash
python main.py input.wcnf --max_time 60 --batch_size 64
```


### Args:

Required:
- input.wcnf: Input [WCNF](https://maxsat-evaluations.github.io/2021/rules.html#input) file .

Optional:
- max_time: (default: 60) Time limit in seconds.
- batch_size: (default: 1) Number of parallel chains.
- heuristic_interval: (default: 1000) how many sampling steps to use the unit propagation heuristic.
- verbose: (default: False) Print progress.
- device: (default: 'cuda' if torch.cuda.is_available() else 'cpu') Device to run on (cuda/cpu).

### TODO:

- Add params for pre-trained RBM model
- Test script
