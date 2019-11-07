##wandb
```
$ pip install wandb
$ wandb login 3b36f172a11223b016bec194a20497bf1c0928af
```
### W&B Imports
```
import wandb
wandb.init(project="ai607", name=exp_name)

# Model instantiation code ...
wandb.watch(model)
```

