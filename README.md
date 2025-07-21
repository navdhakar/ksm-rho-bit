
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