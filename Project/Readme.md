
# A Neural Algorithm of Artistic Style (Gatys et al, 2015)


In this project, I attempt to reproduce the results from the paper A Neural Network of Artistic Style' by Gatys et al. (2015) and explore variations of the model. I combine the content of a photograph with the the style of an art work to produce an artistic inage. 

Below images are used to get the result images. Description of their purpose is as follows. All the images are saved inside Data folder  

#### Neckarfront.jpg -- Used as a content image 
#### The_Starry_Night.jpg --Used as a style image 
#### Der_Schrie.jpg --  Used as another style image.

 I use a pre-trained model - VGG19 'imagenet-vgg-verydeep-19.mat' to get the optimized weights and biases.
 Source of vgg-19 : http://www.vlfeat.org/matconvnet/pretrained/  

#### *style_tranfer.ipynb* - Main Jupyter notebook with the code for our style transfer analysis. Run this notebook to see the results. The entire code can be run in sequence using kernal -> Restart & Run_All

. I experiment with more than one style image to reflect the texture of both the style images in the output image. The section *Extension of the model* in *style_transfer.ipynb* file contains the code used for combnination of the two styles. I used the following util .py files, and imported them in *style_transfer.ipynb*
#### style_util.py-- contains  the functions for style and content loss
#### vgg_util.py --  contains the functions to get the weights and biases of VGG 19 model , and get the layers of the model

Hence, I am optimizing the total_loss twice - to create the result of the original paper-- content + style -- and to create the result of extension of model -- content + style1 +style2

### output2499_1.png -- Output of the first run (content +style)
### output2499_2.png -- Output of the second run (content +style1 + style2)

