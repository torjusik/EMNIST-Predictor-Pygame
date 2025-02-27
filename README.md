# EMNIST Predictor with Pygame
This project utilizes the EMNIST dataset to create a handwritten character recognition system with a graphical interface built using Pygame. Users can draw characters on the screen, and the system will predict the corresponding letter or digit.

### Features
- **Handwritten Character Recognition**: Supports recognition of both letters and digits using the EMNIST dataset.
- **Interactive Drawing Interface**: Draw characters directly on the screen with real-time prediction feedback.
- **Model Training and Evaluation**: Includes scripts to train and evaluate the neural network model on the EMNIST dataset.

## Installation

### Clone the Repository:

```bash
git clone https://github.com/torjusik/EMNIST-Predictor-Pygame.git
cd EMNIST-Predictor-Pygame
```
### Install Dependencies:
Ensure you have Python installed. Install the required packages using pip:

```bash
pip install -r requirements.txt

```
## Usage
### Start the training:
You first need to train a model on the dataset by running agent.py
#### Start the Drawing Interface:
After training a model and saving it, you can run drawing.py to test it out.

## Project Structure
- agent.py: Contains the main logic for handling user input and interfacing with the model.
- drawing.py: Manages the Pygame window and drawing functionalities.
- model_handler.py: Includes functions for loading and saving the neural network model.
- neural_network.py: Defines the architecture of the neural network used for character recognition.

## Acknowledgments
- EMNIST Dataset: Provided by NIST, offering a comprehensive set of handwritten character data.
Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373
- Pygame: A set of Python modules designed for writing video games, used here for creating the interactive interface.
This project combines machine learning and interactive graphics to provide a hands-on experience with handwritten character recognition.
