 # Radar-PointNet
 This repository contains different programs, used for the master thesis "Radar Stereo-Camera Sensor Fusion for 3D Pedestrian Detection"
 # Dataset Generation
 # Frustum-PointNet
 ## YOLO-v3
As the stereo camera has a max range of 20 m, some Pedestrians, which are present on the image, are not present in the stereo camera PC. Therefore, to optimize the training of the 2D object detection, we generated not only 2D bbox for the 3D annotated pedestrians but for all the pedestrians present in the left-image of the SC.
the 2D annotations are present in the file YOLOV3/dataset/RSC/
_Remarque_: the generated 2D annotation are in Kitti format, however the used YOLO version needs that the 2D annotation are in COCO format. the script "" take in charge of transforming the 2D annotation from kitti format to YOLO format.
### train 
the script "" is in charge of training Yolo.v3 through the cmd:
### eval
The script "" outputs a benchmark for different resolution. the results are outputted in the file ""
The script "" is in charge of evaluating and outputting the results for YOLO.v3 through the cmd:
_Remarque_: the results outputted from YOLO.V3 in an XML file. to transform the detection result in KITTI format, 2 methods are employed:
  * As we mentioned, we had to include in the images, the pedestrians that are not included in the SC PC. therefore, we developped the script " " , that permit to filter the 2D detection that corresponds to pedestrians that are 3D annotated. this script work as following:
    
    * load script "" through the cmd " "
    * The image with the 2D ground truth (blue) and numerated 2D detection will appear
    * To delete a 2D detection press, the number of the 2D detection box and then "space"
    * After deleting the unmanted 2D box, press "enter" to show the image without the deleted boxes. if the all the unwanted boxes are all deleted, repress "enter" to move to the next frame. if not, press "r" to restart the deletion process of this frame.
  * the 2D detected pedestrians will automatically be deleted by frustum-Pointnet, As normally there is no PC corresponding to this 2D object detection. the script " " will be in charge of transforming all the 2D detection from XML format to KITTI format.
## Frustum-PointNet
### Train
  the script "" is responsible of training Frutum-Pointnet in either versions (version1 and version 2) through the cmd:
  the scripts will output in the log datei of the corresponding version, a log file which contains quantitaive results on the train and the chosen 2D detection resolution for the val and test dataset for each epoc such as average segmentation accuracy, average detection accuracy per box, average detection accuracy and recall per frame. moreover it will save the 3D detection results for the val and test datase under the file "results_"res".  
### eval
  During training, the trained weights of the last 5 epochs are saved in "". Reeavaluation of the trained Frustum-pointnet can be done through the script cmd:
  this will output the qualitative results in the terminal and update the results in the corresponding "results_res" file.
### test
  Inference is performed through the script:
  
  this will output in the terminal the inference time for each frame and at the the average inference time and the quantitative results.
## Radar-PointNet
### Radar-PointNet-RoI
  This reporsitory allow to train and evalaute Radar-PointNet with the total RoI as input or with the divided RoI in three Anchor as input. the outputs schemas used is the same as explained in Frustum-Pointnet.
  ### Train
      
  ### Eval
  
  ### Test
  
### Radar-PointNEt-Para
  This repository allow to train and evaluate Radar-PointNet with both proposal generation method: iterative method or minima method. MOrover two version exists for this variance: with and without the classification method. for both method we employed a parallelization technique for the segmentation network which cooresponds on grouping the pc for each RoI together. a prallelization technique for the Bbox is employed only whith proposals classification method 
  ### Train:
  for training  Radar-PointNEt-Para, we use a two step training method: first the segmentation with or w/o the proposal classification network are trained and then the bbox network are trained. the next commands represent the train and evaluation cmds for the different existing version:
  _Radar-PointNEt-Para w/o  classification_
  Train the segementation network alone through the cmd:
  
  
  
  This will output the train, val, results in the form of pickle file which contains the frame id,the detected object pc and the RoIs parameters for each frame. The quantitaive results (average segmentation accuracy) are stored in the corresponding logfile. 
  
  Train the bbox network from the corresponding semgentation output is done through the cmd 
  
  Same output schemas is used here as Frustum-PointNet.
  _Radar-PointNEt-Para w/o  classification_
  Train the segmenetation network with the classification network with the cmd:
  
  This will output the train, val, results in the form of pickle file which contains the frame id,the detected object pc present in each proposals and its coresponding classsfication score. The quantitaive results (average segmentation accuracy, average classification network) are stored in the corresponding logfile.
  
  Training the bbox network from the corresponding semgentation output is done through the cmd:
  
  Same output schemas is used here as Frustum-PointNet.
    
  
  
  
    

 
 
 
 
 
 
 
