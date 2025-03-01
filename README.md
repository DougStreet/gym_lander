# 1 Overview
This is a gymnasium lander project, solved with ML and conventionally with PID's. I did this to see how good I could get the PID vs. orders of magnitude more complex NN based solutions.
The PID solution was written before the ML to avoid any "copying" of the solution the ML might find.


# 2 PID Details
This code was written from scratch, and uses 3x separate PID's to control the vertical velocity, horizontal position and angle of the lander.
The tuning was done in about 30 minutes and could likely be improved, but was felt to be acceptable.

## 2.1 Vertical PID
The vertical velocity of the lander is controlled based on a few different things:
1. If the lander is within a certain x-axis window it will fall rapidly using a tiny bit of fuel to assist with control of angle & horizontal position
2. If the lander is outside the same x-axis window, it will fall slowly to allow correction of horizontal position without losing altitude
3. If the lander is close to the ground it will decelerate and fall very slowly to try and "hover slam"

The PID is very stiff with quite a lot of I to remove any steady-state error very quickly. There is no D as we don't tend to overshoot.

## 2.2 Horizontal PID
The horizontal position of the lander is controlled relative to an X-axis coordinate (middle of the landing zone). There is a moderate amount of P, I and D on this controller, where the D is particularly useful to factor horizontal velocity (derivative of error), and it really stops overshoot nicely.

## 2.3 Angle PID
The angle of the lander, trying to maintain 0 angle. It's super simple, using only P and D terms. No I is a good idea as this is a truly proportional problem, and the D stops the lander from wobbling on occasion if the angle is rapidly changing. 

## 2.4 Coupling of Horizontal and Angle PIDs
The horizontal and angle PIDs are coupled by simply summing them together. The angle PID has an output that can swing further than the horizontal to make sure it can always over-ride the horizontal thrusters.

# 3 ML Detials
The ML solution was a copy of https://www.findingtheta.com/blog/solving-gymnasiums-lunar-lander-with-deep-q-learning-dqn with some minor changes.
The number of neurons in layers were changed, as well as the number of layers, all with no real improvements over a 2 hidden layer of 128 neurons.
Additional training episodes were added, upto 5k, but beyond about 3k it's all the same.

Likely, this could be improved with a bit more time (and knowledge), especially under worst-case runs where the ML model occasionally crashes.

# 4 Results

## 4.1 PID Results
Here is a video of the PID solution:
![](https://github.com/DougStreet/gym_lander/blob/main/videos/pid-video.gif)

Average Score: 276.48 Max Score: 313.30 Min Score: 231.45

## 4.2 ML Results
Here is a video of the best model I could wrangle:
![](https://github.com/DougStreet/gym_lander/blob/main/videos/rl-video.gif)

I captured a few of the results from different sizes of networks:

2x 128 neuron hidden layers - Average Score:  281.33 Max Score:  317.25 Min Score 60.57

2x 256 neuron hidden layers - Average Score:  275.39 Max Score:  320.31 Min Score -116.31

4x 128 neuron hidden layers - Average Score:  283.28 Max Score:  327.43 Min Score 16.09