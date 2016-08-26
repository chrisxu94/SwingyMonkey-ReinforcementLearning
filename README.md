# SwingyMonkey-ReinforcementLearning

## Python implementation of MDP Reinforcement Learning to learn how to play the Swingy Monkey game (essentially Flappy Bird).

One cool twist is that we implement feature-based learning, as opposed to game-state-based learning. That is, we implement 
online learning on specified game features (like vertical distance from monkey to ground, horizontal distance to nearest 
branch, etc.) Traditionally, MDP's are solved by learning the values/policies on the all the possible states. Since there
are comparatively so many more possible states in the game, state-based learning would have taken much longer than 
feature-based learning.


