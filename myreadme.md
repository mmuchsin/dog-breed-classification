Project Set Up and Data

The submitted project contains a README file that contains:

A short introduction of the project
Project Setup Instructions
Explanations of the different files used in the project
Code sample for querying a model endpoint
Explain insights from the model
At least 2 relevant and informative images, terminal outputs, or screenshots

Hyperparameter Optimization
The submitted README file contains a screenshot of your hyperparameter tuning job that shows at least 2 different training runs and their results.

Profiler and Debugger
The Jupyter notebook or the README should include a line plot showing the status of a variable throughout the training process.
In case the plot shows an anomalous behavior, the Jupyter notebook or README should include the steps they took to debug it. If not, it should include the steps they would have to take to debug an error.

Model Deployment
The README should also contain a screenshot showing an active endpoint in SageMaker

# Dog Breed Image Classifications with Pytorch (ResNet50) and AWS Sagemaker

The objective of this project is to create a deep learning model that is capable of identifying 133 different breeds of dogs. This project will use [Pytorch](https://pytorch.org/) as a deep learning framework and [Resnet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) as a pre-trained model. The dataset used can be accessed from [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). This project will outlines the process for an end-to-end image classification project, utilizing the resources and capabilities of AWS Sagemaker.

## Project Setup
1. Login to your AWS account and launch the sagemaker studio.
2. Clone this [repo](https://github.com/udacity/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter) as the starting point.
3. Complete the `hpo.py` file. It's required to do hyperparameter tuning.
4. Complete the `train_model.py` file. It's required to do Debugging and Profiling.
5. Create an `inference.py` file. It's required to deploy an endpoint.
6. Open train_and_deploy.ipynb
   >I use Python 3 (PyTorch 1.12 Python 3.8 GPU Optimized) kernel with pre-installed PyTorch libraries and ml.m3.medium as the cheapest instance just for running the notebook
7. Install dependencies
   ```
   pip install -Uqq pip awscli sagemaker smdebug
   ```
8. Download the dataset
   ```
   wget -nc https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
   ```  

## Project Pipeline
<div align='left'>
   <img src="./src/img/1-project-diagrams.png" alt="pipeline"style="width: 350px;"/>
</div>
<div>
   <em>image by udacity</em>
<div>



## Project Overview

### Dataset
The datasets have been split to train, validation, and test set. The train set contains 6680 images divided into 133 classes corresponding to the dog breeds. The validation set contains 835 images and the test set contains 836 images. Both the validation set and test set have images for each class to check model performance.

### Hyperparameter Tuning
- screenshot
<div align='left'>
   <img src="src/img/2-hpo_crop.png" alt="hpo"style="width: 720;"/>
</div>
- top 3 training job
  
| batch_size | epochs | gamma | learning_rate | step_size | TrainingJobName | TrainingJobStatus | FinalObjectiveValue | TrainingStartTime | TrainingEndTime | TrainingElapsedTimeSeconds |  |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| "32" | 6.0 | "0.5" | 0.016581 | 1.0 | dog-breed-clf-hpo-230113-0640-018-9346f6ce | Completed | 0.8754 | 2023-01-13 08:14:08+00:00 | 2023-01-13 08:22:36+00:00 | 508.0 |  |
| "64" | 21.0 | "0.5" | 0.002352 | 3.0 | dog-breed-clf-hpo-230113-0640-011-e6297d35 | Completed | 0.8743 | 2023-01-13 07:30:34+00:00 | 2023-01-13 07:56:12+00:00 | 1538.0 |  |
| "256" | 21.0 | "0.25" | 0.015431 | 3.0 | dog-breed-clf-hpo-230113-0640-016-5401b3c0 | Completed | 0.8659 | 2023-01-13 08:00:47+00:00 | 2023-01-13 08:27:09+00:00 | 1582.0 |  |

- best hyperparameters:
```python
{
   'batch_size': '32',
   'epochs': '6',
   'gamma': '0.5',
   'step_size': '1'
}
```
### Debugging and Profiling
![dp]()
### Model Deployment
![endpoint](src/img/3-endpoint_crop.png)