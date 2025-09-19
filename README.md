# Federated Learning Potential Game

This repository contains the implementation code for the paper *"A Potential Game Perspective in Federated Learning."*

First, run `main_game.py` to compute the Nash equilibrium of the FL-Game under different reward factors.

```bash
python main_game.py --m 1000
```

The four cases involving critical reward factors will be saved in the `.\utils` directory.

Then, to evaluate the FL training performance, execute `main_train.py` with the `case` argument from 1 to 4.

```bash
python main_train.py --m 1000 --case 1
```

All results and plots will be saved in the `results` folder.
