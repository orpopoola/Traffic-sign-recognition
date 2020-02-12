# **Traffic Sign Recognition** 

## Writeup

### This writeup presents the methods, challenges and process used to complete the Traffic Sign Recognition project by implementing convolutional neural networks for prediction.

---
**Traffic Sign Recognition** The main project is presented in the Traffic_Sign_Classifier.ipynb file and the Traffic_Sign_Classifier.html file

**Process to build a Traffic Sign Recognition System** To successfully complete the project, I followed the objectives given as a guide. I also made use of provided software snippets and a modified LeNet convolutional neural network architecture.

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43
In this work, the validation set was provided in the initial data I downloaded. Therefore, I did not split the training data to get a validation set rather, I just loaded the validation set provided.

#### 2. Include an exploratory visualization of the dataset.

In the notebook, I included a visualization histogram of the spread of the classes.
### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I tried the use of grayscale and normalization, as advised in the work, I used normalization by computing the mean and standard deviation of the dataset then I normalized the data by dividing the deviation from the mean by the standard deviation. After this, I checked and the mean of the new data was 0 (python displayed it as something 4.3 x10^-15 which is very close to zero and the standard deviation was computed as 1.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 26x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x16  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten  	 		 	|			outputs 1x1x400 					|
| Fully connected		||			outputs 1x1x120 					|
| Dropouts				|												|
| Fully connected		||			outputs 1x1x84  					|
| RELU					|												|
| Fully connected		||			outputs 1x1x43  					|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer which is better than the gradient descent, I used a learning rate of 0.0005, a batch size of 128 and 15 epochs. I avoided increasing the number of epochs unnessarily to prevent data overfitting.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 94%
* test set accuracy of 93%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.


The third image was difficult to classify because it is not a part of the training set or classification label. However, the network was able to predict it very well.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Go straight or right	| Go straight or right							|
| Turn left 			| Turn left ahead								| **there was no turn left in the training set**
| No passing	   		| No Passing					 				|
| Yield     			| Yield             							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93%. However, it could be reasonably assumed that the 'turn left ahead' prediction was very good for the 'turn left' sign which will make the accuracy 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for the softmax function and the code for making predications are in the 62nd and 66th cell. These were used to write probabilities of the signs.
The probabilities for the signs were predicted 


Test 0 
 Top 5 Probability: [  9.99999285e-01   6.13574912e-07   1.07986821e-07   2.49230734e-08
   1.37940255e-08] 
 Top 5 classes: [14 25  1 29 22]
Test 1 
 Top 5 Probability: [  9.99925017e-01   7.28122322e-05   2.14594183e-06   1.82680704e-10
   7.06582848e-11] 
 Top 5 classes: [36 34 35 38 33]
Test 2 
 Top 5 Probability: [  9.99427199e-01   5.27567172e-04   2.26389711e-05   2.12529485e-05
   1.41157011e-06] 
 Top 5 classes: [34 31 12 33 39]
Test 3 
 Top 5 Probability: [  1.00000000e+00   1.83532253e-14   9.70139551e-16   7.27413472e-16
   4.97279626e-20] 
 Top 5 classes: [ 9 10 16 12 13]
Test 4 
 Top 5 Probability: [  1.00000000e+00   6.14012463e-17   2.06882824e-22   4.03826851e-24
   2.07221145e-27] 
 Top 5 classes: [13  9 15  3 39]



