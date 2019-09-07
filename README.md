# FindWaldo
Have you ever struggled to find Waldo? This agent does it effortlessly for you!

## Installation
Clone the repository
```bash
git clone https://github.com/Agnar22/FindWaldo.git
```

navigate into the project folder
```bash
cd FindWaldo
```

if everything went well, you should now be able to run the code
```bash
python3 kerasfindwaldo.py
```

## Motivation
The ultimate goal of this project was to have an AI that was able to mark Waldo on the images. 

## Method
The naive approach to this problem is supervised learning with conv-nets: you label a bunch of "finding Waldo" images and split the data in a test- and training set, etc. The problem with this approach is that NN works best with large amounts of data and there are a limited number of "finding Waldo" images. Additionally the time required to label all those images is substantial. Thus this does not look like a feasible solution.

Despite these problems, the approach taken in this repo was quite similar to the supervised learning approach described above. However, the data labeling and the construction of the neural network was carried out differently, and exactly how this was done is what I am going to explain to you next.

__Data labeling__

First and foremost, 59 finding waldo images where marked like this __Fig__ and erased from the image __fig__. Then, by carefully extracting his head you can now easily generate a lot of fake "finding Waldo" images.

The idea is that there are always one recurring element in the "finding Waldo" images: they all contain Waldos head! This might seem obious, but the crucial part is that the rest of him is not always shown in the images, thus it is not useful to mark his entire body. Marking his entire body could also make the neural network overfit due to the fact that in the images where you see his entire body, he is often not wearing the same clothes __fig__ __fig__.

To avoid overfitting, his head was randomly tilted, scaled and placed on a background. By having half of the images without his head, we now have a method of generating a large dataset of labeled "finding Waldo" images without too much trouble.



__The Neural Network__

The problem with a standard convolutional neural network in this context __fig__ is that the input must be of a fixed size. This poses a problem for us because our "finding Waldo" images are of different sizes. To solve this, the fully connected layers were converted to convolutional layers. *Say whaaat!?* This is all explained in [this article](http://cs231n.github.io/convolutional-networks/#convert "CS231n Convolutional Neural Networks for Visual Recognition") from Stanford. The advantage if this is that the network now accepts all images that have dimentions 64x64 pixels or larger. This allows us to have 64x64 images to train on, and be able to scale it up to larger images later on when we want the agent to really find Waldo. Thus the task of finding Waldo has now been reduced to a binary classification task for images. *Whew.*


## Results

When measuring training- and testing results it is important to have a clear boundary between what is considered training- and testing data. The raw images where divided into two groups where one of the groups were used to generate the training data (extracting Waldo heads and pasting them on random 64x64 cutouts of the original images, as described under __data labeling__) while the other were used to demonstrate the accuracy of the agent. This separation is vital, as a leakage of the same Waldo heads from training to testing data would give a false sense of accuracy.

Here are the predictions on some of the raw images that were not used to generate training data:
<p align='center'>
<img width="50%" src="https://github.com/Agnar22/FindWaldo/blob/master/READMEImages/5.jpg"><br>
<b>Figure 6</b>: the agent is able to find Waldo in this image
</p><br><br>
<p align='center'>
<img width="50%" src="https://github.com/Agnar22/FindWaldo/blob/master/READMEImages/10.jpg"><br>
<b>Figure 7</b>: here it actually finds Walda, Waldo is down to the left.
</p><br><br>
<p align='center'>
<img width="50%" src="https://github.com/Agnar22/FindWaldo/blob/master/READMEImages/8.jpg"><br>
<b>Figure 8</b>: several persons are marked here; Waldo, Walda and some of the kids.<br> This is not a bug, its a feature!
</p><br><br>
<p align='center'>
<img width="50%" src="https://github.com/Agnar22/FindWaldo/blob/master/Data/Raw/Test/16t.jpg"><br>
<img width="50%" src="https://github.com/Agnar22/FindWaldo/blob/master/READMEImages/16t.jpg"><br>
<b>Figure 9</b>: an image with only one real Waldo and the corresponding heatmap<br> from the agent.
</p>
The Waldo image from <b>Figure 9</b> is an interesting one; there are many Waldo lookalikes, but only one real (as described in the text on the image). The agent obviously did not find him because he is not trained to do that in this manner, but we clearly see that it marks the Waldos. Additionally, looking at the stairs in the middle on the heatmap, we see an interesting pattern; the Waldos that take of their hats are not recognized by the agent. This is of course a direct consequence of the training, where the agent is only trained to find Waldo wearing a hat.

As you can see the agent is able to find Waldo or something that might resemble him, therefore I would call it a success.


## Other resources
* To get a real grip around convolutional neural networks, I recommend [this medium article](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53 "A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way").
* I also recommend reading the [article](http://cs231n.github.io/convolutional-networks/ "CS231n Convolutional Neural Networks for Visual Recognition") from Stanford that I reffered to earlier in this README.

## License
This project is licensed under the MIT License.
