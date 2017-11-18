# Pytorch-Double-DQN
This is project is a PyTorch implementation of [Human-level control through deep reinforcement learning] along with the Double DQN improvement suggested in [Deep Reinforcement Learning with Double Q-learning].

# Requirements
- python 3.5
- [pytorch]
- [pyyaml]
- [gym]

# Usage
```sh
python main.py [-h] --config PATH --save_dir PATH [--modelpath PATH]
```
The config file is a yaml file used to provide arguments include mode (train or eval). A sample config file has been provided [here].
# References
- https://github.com/transedward/pytorch-dqn
- https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3

[Human-level control through deep reinforcement learning]: http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
[Deep Reinforcement Learning with Double Q-learning]: https://arxiv.org/abs/1509.06461
[pyyaml]: https://anaconda.org/anaconda/pyyaml
[gym]: https://github.com/openai/gym#installation
[pytorch]: http://pytorch.org/
[here]: https://github.com/Shivanshu-Gupta/Pytorch-Double-DQN/blob/master/config.yaml
