<h1 align="center">
  SuperMarioAI
</h1>
<p align="center">
  <a href="https://github.com/erickfunier">
    <img alt="Made by Erick Santos" src="https://img.shields.io/badge/made%20by-Erick%20Santos-lightgrey">
  </a>
</p>
<p>
This code contains a Player created with Machine Learning and Neural Network who can play the state YoshiIsland2 from Super Mario World - SNES.
</p>
<p>
The neat takes just one generation of genoma, with 150 genomas. The winner from Gen 1 was the id 68.
</p>
<p>
The folder train-codes contains all saves from neat, but you can train changing the parameters to improve the results.
</p>

<h2>
  Running the application
</h2>
<p>
  To run the winner and the application you can follow the below instructions:
</p>

<h2>
  Dependencies
</h2>
<p>
  Python 3.7
</p>
<p>
  gym-retro
</p>
<p>
  Gym 0.21.0
</p>
<p>
  ImportLib Metadata 4.8.1
</p>

<h2>
  Install Instructions
</h2>
<p>
  Uses Python 3.7
</p>
<p>
  Install <a href="https://retro.readthedocs.io/en/latest/getting_started.html">Gym Retro</a>
</p>

    pip install gym-retro
    
<p>
  Install Gym version 0.21.0
</p>

    pip install gym==0.21.0

<p>
  Install ImportLib Metadata 4.8.1
</p>

    pip install importlib-metadata==4.8.1

<p>
  Install OpenCV
</p>
    
    pip install opencv-python

<p>
  Install Neat
</p>

    pip install neat-python

<p>
  Import the ROM to your Gym Retro
</p>

    python -m retro.import <completed-folder-rom>
    
<p>
  Example:
</p>
    
    python -m retro.import "C:\Users\Erick\Desktop\Super Mario Brother\rom"
  
<h2>
  Running the application
</h2>
<p>
  Run the python play.py to view the gameplay
</p>

    python play.py
    
<p>
  Enjoy
</p>
