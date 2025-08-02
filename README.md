
```bash
python -m train
```
```bash
# You can override default arguments like this
python -m train max_steps=100
```
7. To see the TensorBoard logs for all your runs:
```bash
tensorboard --logdir humanoid_walking_task
```
8. To view your trained checkpoint in the interactive viewer:
- Use the mouse to move the camera around
- Hold `Ctrl` and double click to select a body on the robot, and then left or right click to apply forces to it.
```bash
python -m train run_mode=view load_from_ckpt_path=humanoid_walking_task/run_<number>/checkpoints/ckpt.bin
```
**Using old Ksim lib to render on wsl2 ubuntu new one use Qt which is not properly supported in windows WSL2

make necessary changes in ksim-0.1.2 and install using:

1. go to the code dir
```bash
cd ksim-0.1.2
```

2.install the latest change package
```bash
pip install -e .
```


## Implemented Simple PPO:
Reason: Ksim training was acting weird on left arm for g1 robot could not figure out why

- using GruCell RNN, need to work on better reward function, no loss viz yet to asses the trainings

- actor: 256 hidden state, 5 layer deep

- critic: 256 hidden state, 5 layer deep

1. Train
```bash
python simple_ppo.py --max_steps 10
```

2. Test
```bash
python simple_ppo.py --task test --run_mode view  --load_checkpoint=training_runs/run_5/ppo_humanoid_final.pth
```

## Working on PPO V2
added a loss and training vizualation

- Training with live plots
```bash
python main.py --task train --enable_live_plots --max_steps 100000
```

- Testing with visualization
```bash
python main.py --task test --test_episodes 10 --run_mode view
```

- Analyze completed run
```bash
python main.py --task analyze --load_checkpoint training_runs/run_1/ppo_humanoid_final.pth
```

- Compare multiple runs
```bash
python main.py --task compare --run_dirs training_runs/run_1 training_runs/run_2 --run_names "Baseline" "Tuned"
```