# Policy-Evaluation
Policy evaluation is the building block of "Policy Iteration" Algorithm. This algorithm is used frequently in Reinforcement Learning field.

Small Gridworld is a rectangular shaped grid. Which has two terminal states. Out goal is to find an optimal policy to reach terminal policy.
This problem has been taken from the famous book "Reinforcement Learning: An Introduction" by Richard S. Sutton & Andrew G. Barto. You can find the problem in Example 4.1 from the second edition of the book.
For better understading, I suggest to go through [lectures by David Silver](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

To evaluate a policy, we will first choose an arbitary value function(we will fill the matrix with zeros). 
Then using Bellman update equation, we will update each state's value function.
After significant number of iterations, we can see a good optimal policy.

In policy iteration algorithm, all the states are assigned random policies at first.
Depending on those policies, we have to calculate value function and have to find maximum value from 4 adjacent states. 
Then, according to highest value found in previous step, we will update our policy(direction to that state).
