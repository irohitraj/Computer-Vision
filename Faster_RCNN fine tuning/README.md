# Fine tuning Faster RCNN
This repo contains the code for fine-tuning Faster RCnn on your custom dataset. You can use any of the jupyter notebook you want to do this task.
1. [Notebook_1]([https://github.com/irohitraj/Computer-Vision/blob/main/Faster_RCNN%20fine%20tuning/Supervised%20Fine-tuning%20using%20class.ipynb]) Notebook contains detailed commented code using class
2. Fine-tuning using just functions. Basic structure while learning


### Dataset:
  1. Dataset has been generated manually using VGG annotator present in the zip file
  2. How to create a dataset? Follow the steps below:
     a. Extract the zip on your local machine and open via.html file
     b. Now name your project. After that add your images by clicking on Add Files tab
     c. now expand Attribute tab and give a name, ex: obj_classes and then click on "+" symbol
     d. Now fromt the new pane that opens below it, change "Type" from text to "dropdown"
     e. Now give the class label and the class name
     f. Now scroll up and select rectangle from "Region Shapes". Mark the intended obj in rectangle and give it it's name
     g. Now click on "Annotation" on top and export in "COCO" Annotation format 
### Environment Setup:        
  1. create and activate your virtual environment
  2. Install pytorch with cuda: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  3. pip install pycocotools
  4. pip install jupyter notebook

### References:
  1. ### Youtube tutorial: https://youtu.be/xYd95gppJ-0
  2. ### Youtube tutorial: (https://www.youtube.com/watch?v=YA7BqiUTCwM&t=22s)
  3. ### VGG Annotator (https://www.robots.ox.ac.uk/~vgg/software/via/)
