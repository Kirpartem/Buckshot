# Buckshot Roulette Agent Mastermind

This project is a complete implementation for training an Agent to master Buckshot Roulette via Self-Play.

## Features:

- Complete game implementation. Most rules are 1-1 with the real game, some niche interactions (e.g predetermined amount of lives/blanks based on number of batches) may differ, though not enough to matter. Optimized for very high speed.

- Arena setup for Self-Play "Tournament" curriculum against previous versions. This is miles better than normal self-play for this use case.

- Configurable training config. Currently set to late training for improved convergence. If you want to start from scratch, I highly recommend you change its values.

- Converter script to quickly convert the model to .onnx for web implementation.

## Does not feature:

- Rendering loop. Removed because it was sloppy. If you want, you can find it in commit history.

## Scripts:

### Installing requirements with pip:

`pip install -r requirements.txt`

### Training:

`python -m agent.train`

Pretty self-explanatory. If you want to start from scratch, delete the agent/models folder.

### Converting to .onnx

`python -m converter`

Yup, this one is pretty self-explanatory too.

# Important notice:

I do not own the original Buckshot Roulette game. The game is available on [Steam](https://store.steampowered.com/app/2835570/Buckshot_Roulette/). I do not own any assets from the game, and neither do I use them. Props to Mike Klubnika and Critical Reflex for making and publishing this awesome game.
