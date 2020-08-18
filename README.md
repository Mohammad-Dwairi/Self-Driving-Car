# Self-Driving-Car
Self Driving Car simulation built with Tensorflow and Keras neural network library.
The model is connected to Udacity Self Driving Car Simulator using Flask framework and socketio.



![alt text](https://i.imgur.com/cWUbB2i.png)


# Model
The Neural Network Model follows the Nvidia Model Architecture which consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers. The input image is split into YUV planes and passed to the network:



![alt text](https://i.imgur.com/2KHVCd2.png)





# Steering Angles Histogram:

![alt text](https://i.imgur.com/uOXkYDi.png)

# Balanced Steering Angles Histogram:
The training data is skewed towards the middle because most of the time the car is driven in a straight line while training.
If we train the convolutional neural network based on this data, the model could become biased towards driving straight all the time.
Solution: Flatten the data distribution and cut off extraneous samples for specific bins whose frequency exceed 400.
![alt text](https://i.imgur.com/zDSQNv5.png)
