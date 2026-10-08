**EMNIST Handwritten Character Recognizer (from scratch)**

A neural network that recognizes handwritten digits and letters, written in plain NumPy with no machine-learning frameworks. It is trained on the EMNIST Balanced dataset (47 classes) and reaches about 70% accuracy on held-out data.

I started from the Vizuara "neural network from scratch" tutorial on YouTube, which builds a digit classifier on MNIST, and extended it to cover digits and letters together on EMNIST. The code in this repo is my own adaptation and extensions.

**RESULTS**
*Metric | Value*

Training Accuracy (1000 Iterations) | ~71%
Dev Accuracy (20% held-out split) | ~70%
Model | 784 inputs --> 128 --> 47 softmax

**Training Curves can be viewed in the graphs folder.**
*What do the curves show?*

Training and dev accuracy stay within about two points of each other, so the model is not overfitting. It is limited by its size and by plain full-batch gradient descent, not by memorization.
Loss starts near 13.8, far above the ~3.85 expected from random guessing across 47 classes. My uniform (-0.5, 0.5) weight initialization starts the network confidently wrong, and the first ~50 iterations mostly undo that.
Both curves are still improving slowly at iteration 1000.

**How it works**
*Data*
EMNIST Balanced has 47 classes: the 10 digits, 26 uppercase letters, and 11 lowercase letters that look different from their capitals (a, b, d, e, f, g, h, n, q, r, t). The other lowercase letters are merged into their uppercase class. Classes are balanced, with 2,400 training images per class.

*Preprocessing* (importdata.py)
EMNIST images are stored transposed, so each image is transposed back to upright before training.
Rows are shuffled with a fixed seed and split 80/20 into training and dev sets. The fixed seed means the training and evaluation scripts always build the same split.
Pixel values are scaled from 0-255 to 0-1.

*Model and Training* (trainingletters.py)
Two layers: a 128-unit ReLU hidden layer and a 47-way softmax output.
Forward propagation, backpropagation, and gradient descent are all implemented by hand.
Full-batch gradient descent, learning rate 0.2, 1000 iterations.
Training and dev accuracy and training loss are logged every 50 iterations, and the curves are saved as PNGs in graphs/.

*Prediction* (predict_image.py)
Flattens transparency, converts to grayscale, and inverts dark-on-light images to match EMNIST's white-on-black style.
Crops to the character, pads to a square, shrinks, and centers it in a 28x28 frame so a drawing is framed like a training image.
Prints the top 5 predictions and shows what the model saw.

**How can you do this?**
*Get the data*
Download the EMNIST dataset from Kaggle (see also the official NIST page) and place these files in the data/ folder. Rename the file emnist-balanced-train.csv to training.csv (its path should be data/training.csv). This file is not included in this repo due to its size.

*Train*
Run trainingletters.py. NOTE: this is reduntant, as training parameters are already provived in emnist_params.npz. DOUBLE NOTE: this can take HOURS.

*Hold-Out Set*
Run display.py. Shows one dev-set character with the predicted and actual labels, then prints the accuracy on the whole dev set.

*Predict with your own image*
Run predict_image.py in your terminal using the following bash command:
python predict_image.py path/to/your_character.png
Where path/to/your_character.png is the file name for the image you wish to scan through the network. Works best with a single character, written large, with a plain background.

**Limitations**
About 70% accuracy is well below what convolutional networks reach on this dataset. Some errors are unavoidable because characters like O and 0 or I, l and 1 are genuinely ambiguous.
Merged classes mean the model can't tell some uppercase and lowercase letters apart.
Real handwriting from a mouse or a phone can look different from EMNIST scans. The preprocessing in predict_image.py is a heuristic, and its margin and threshold values were chosen by hand rather than tuned.
Only single characters are supported so far.

**Acknowledgements**
Tutorial: Vizuara, "Neural Network from Scratch" on YouTube.
Dataset: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters.
