```
ValueError: ('Expected `model` argument to be a `Model` instance, got ', None)
```
Atari Game
1. Clone the Git: git clone	git clone https://github.com/reinwzhang/Atari.git
2. Command Line: cd Atari
3. Command Line: pip install -r requirement.txt
4. pip install gym[atari]
5. install ffmpeg:  https://www.ffmpeg.org/download.html
6. Train: python dqn_atari.py -modeltype dueling_dqn --dueling_type avg ; You can choose different modeltype to train: dqn, double_dqn, dueling_dqn
7. Test: python dqn_atari.py --mode test --modeltype dueling_dqn --dueling_type avg --model_path "Path_to_your_model_weigthts"

