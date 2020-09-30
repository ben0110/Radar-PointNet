 # Radar-PointNet
 This repository contains different programs, used for the master thesis "Radar Stereo-Camera Sensor Fusion for 3D Pedestrian Detection"
 # RSC-Dataset Generation
 ## Data collection
 as shown in the figure below, data collection occur through starting the rospackages of each sensor and storing the published data in a bagfile.
 
 ![teaser](https://github.com/ben0110/Radar-PointNet/blob/master/pictures/data_col_pip.jpg)
 
 ### ZED Camera
 starting publishing data  with the ZED camera occures through the cmd:

    roslaunch zed_display_rviz display_zed.launch 

 adjusting the camera parameters can be done through the file 
 #### Radar AWR1843-boost
 starting publishing data with the  AWR1843camera occures through the cmd:
     
     roslaunch ti_mmwave_rospkg 1843es1_long_range_wo_rviz.launch
 
 adjusting the radar parameters can be don through the file
 
 ### DCA10000-evm (optional) 
 starting the DCA1000 evm occures through the cmd:
 
    rosrun dca1000data_collection DCA1000_cli.py
 
 to start publishing data, click the button "capture" on the DCA1000-evm board.
 
 ### data collection with rosbag:
 
 storing the data published from the different boards is done through this command: 
    
    rosbag record /zed/zed_node/right/camera_info /zed/zed_node/right_raw/image_raw_color /zed/zed_node/point_cloud/cloud_registered /zed/zed_node/left_raw/camera_info /zed/zed_node/left_raw/image_raw_color /ti_mmwave/radar_scan_pcl /ti_mmwave/radar_scan /DCA1000_rawdata --duration=5 -b 3048

 ## data synchronization and extraction
 
 We stored the collected data under the "date of capture" folder. Therefore, The data extraction and synchronization process occurs through the cmd
   
    bash extract.sh <data-collection-date> <bag-name>
 
 this script perfromes time and spatial synchronization between the data and storing the sensor data in their respective files under the rosbag name.(example figure)
 
## data annotation

The data annotation occures through the 3d-BAT application.
The data, which is extracted from a bag has to be first copied in the 3D-BAT folder. After performing annotation, the annotation file has to be downloaded and copied to the respective bag file.

## dataset generation
 
 the dataset generation can be applied on the whole annotated data bags with the script or through adding a specifig annotated data bag to the dataset through the cmd:
         
## dividing the dataset:
the actual dataset was generated through dividing the generated frames randomly for the train and val dataset for the frames 0..3145. the test dataset was picked per hand from the frames [150]
Remarque: from the [], only the annotations of the frames used for the test dataset, were rechecked. the other part still has to be rechecked
Another method to generate the dataset can be done through the script, were it is based on the meta data stored in the excel file () ...  
 # Frustum-PointNet
 ## YOLO-v3
As the stereo camera has a max range of 20 m, some Pedestrians, which are present on the image, are not present in the stereo camera PC. Therefore, to optimize the training of the 2D object detection, we generated not only 2D bbox for the 3D annotated pedestrians but for all the pedestrians present in the left-image of the SC.
the 2D annotations are present in the file YOLOV3/dataset/RSC/
_Remarque_: the generated 2D annotation are in Kitti format, however the used YOLO version needs that the 2D annotation are in COCO format. the script "" take in charge of transforming the 2D annotation from kitti format to YOLO format.
### train 
Training Yolo.v3 occurs through the cmd:
      
     python3 train.py --epochs 251 --cfg cfg/kitti.cfg --data data/RSC.data --multi-scale --weights weights/yolov3.pt --cache-images --device 0 

### eval
The cmd below outputs a benchmark for different resolution. the results are outputted in the txt file benchmark.txt

    python3 test.py  --data data/RSC.data --weights weights/best.pt --cfg cfg/kitti.cfg --save-json --device 0 --batch-size 1 --task benchmark

The command below is in charge of evaluating and outputting the results for YOLO.v3 for chosen resolution through the cmd:
   
    python3 test.py  --data data/RSC.data --weights weights/best.pt --cfg cfg/kitti.cfg --save-json --device 0 --batch-size 1 --img-size <res>
 
_Remarque_: the results outputted from YOLO.V3 are in an json file. To transform the detection result in KITTI format, 2 methods are employed:
  * As we mentioned, we had to include in the images, the pedestrians that are not included in the SC PC. therefore, we developped the script "Yolov3_val_results.py" , that permit to filter the 2D detection that corresponds to pedestrians that are 3D annotated per hand. this script work as following:
    
    * load the script through the cmd:
      
         python3 Yolov3_val_results.py <res>
       
    * The image with the 2D ground truth (blue) and numerated 2D detection will appear
    * To delete a 2D detection press, the number of the 2D detection box and then "space"
    * After deleting the unmanted 2D box, press "enter" to show the image without the deleted boxes. if the all the unwanted boxes are all deleted, repress "enter" to move to the next frame. if not, press "r" to restart the deletion process of this frame.
  * the 2D detected pedestrians will automatically be deleted by frustum-Pointnet, As normally there is no PC corresponding to this 2D object detection. the script "Yolov3_val_results_auto.py" will be in charge of transforming all the 2D detection from json format to KITTI format from the cmd:
  
         python3 Yolov3_val_result_auto.py <res>
  
  
## Frustum-PointNet

![teaser](https://github.com/ben0110/Radar-PointNet/blob/master/pictures/F_tnet_arch.jpg)


### Train
  the script "train.py" is responsible of training Frutum-Pointnet in either versions (version1 and version 2) through the cmd:
  
    python train/train.py --gpu 1 --model frustum_pointnets_v2  --log_dir train/log_v2 --num_point 3500 --max_epoch 201 --batch_size 32 --decay_step 800000 --decay_rate 0.5  --res <res> --data_path /root/frustum-pointnets_RSC/dataset/
  
  the scripts will output in the log datei of the corresponding version, a log file which contains quantitaive results on the train and the chosen 2D detection resolution for the val and test dataset for each epoc such as average segmentation accuracy, average detection accuracy per box, average detection accuracy and recall per frame. moreover it will save the 3D detection results for the val and test datase under the file "results_"res".  
### eval
  During training, the trained weights of the last 5 epochs are saved in "". Reeavaluation of the trained Frustum-pointnet can be done through the script cmd:
  this will output the qualitative results in the terminal and update the results in the corresponding "results_res" file.
   
    python train/eval.py --gpu 1 --model frustum_pointnets_v2  --log_dir train/log_v2 --num_point 3500 --max_epoch 201 --batch_size 32 --decay_step 800000 --decay_rate 0.5  --res 204 --data_path /root/frustum-pointnets_RSC/dataset/
### test
  Inference is performed through the cmd:
  
     python train/test.py --gpu 1 --model frustum_pointnets_v2  --log_dir train/log_v2 --num_point 3500 --max_epoch 201 --batch_size 32 --decay_step 800000 --decay_rate 0.5  --res 204 --data_path /root/frustum-pointnets_RSC/dataset/
  
  this will output in the terminal the inference time for each frame and at the the average inference time and the quantitative results.
## Radar-PointNet
### Radar-PointNet-RoI

![teaser](https://github.com/ben0110/Radar-PointNet/blob/master/pictures/Radar-Pointnet.jpg)


  This reporsitory allow to train and evalaute Radar-PointNet with the total RoI as input or with the divided RoI in three Anchor as input. the outputs schemas used is the same as explained in Frustum-Pointnet.
  ### Train
    python train/train.py --gpu 1 --model frustum_pointnets_v2  --log_dir train/log_v2 --num_point 512 --max_epoch 11 --batch_size 32 --decay_step 800000 --decay_rate 0.5  --AB_proposals True/False
      
  ### Eval
    python train/eval.py --gpu 1 --model frustum_pointnets_v2  --log_dir train/log_v2 --num_point 1500 --max_epoch 1 --batch_size 32 --decay_step 800000 --decay_rate 0.5  --restore_model_path  /root/frustum-pointnets_RSC_RADAR/train/log_v2/10-05-2020-13:59:22/ckpt/model_200.ckpt  --AB_proposals True/False
  
  ### Test
    python train/test.py --gpu 1 --model frustum_pointnets_v2  --log_dir train/log_v2 --num_point 1500 --max_epoch 1 --batch_size 32 --decay_step 800000 --decay_rate 0.5  --model_path  /root/frustum-pointnets_RSC_RADAR/train/log_v2/10-05-2020-13:59:22/ckpt/model_200.ckpt --AB_proposals True/False
  
### Radar-PointNEt-Para

  This repository allow to train and evaluate Radar-PointNet with both proposal generation method: iterative method or minima method. MOrover two version exists for this variance: with and without the classification method. for both method we employed a parallelization technique for the segmentation network which cooresponds on grouping the pc for each RoI together. a prallelization technique for the Bbox is employed only whith proposals classification method 
  ### Train:
  for training  Radar-PointNEt-Para, we use a two step training method: first the segmentation with or w/o the proposal classification network are trained and then the bbox network are trained. the next commands represent the train and evaluation cmds for the different existing version:
  _Radar-PointNEt-Para w/o  classification_
  
![teaser](https://github.com/ben0110/Radar-PointNet/blob/master/pictures/RADAR-pointnet-para.jpg)
  
  Train the segementation network alone through the cmd:
  
    python train/train_seg.py --gpu 1 --model frustum_pointnets_seg_v2  --log_dir train/log_v2 --num_point 25000 --max_epoch 201 --batch_size 4 --decay_step 800000 --decay_rate 0.5  --data_path /root/frustum-pointnets_RSC/dataset/ --pkl_output_path /root/frustum-pointnets_RSC_RADAR_fil_PC_batch_para/dataset/RSC/seg/
  
  
  This will output the train, val, results in the form of pickle file which contains the frame id,the detected object pc and the RoIs parameters for each frame. The quantitaive results (average segmentation accuracy) are stored in the corresponding logfile. 
  
  Train the bbox network from the corresponding semgentation output is done through the cmd:
  
    python train/train_bbox.py --gpu 1 --model frustum_pointnets_bbox_v2  --log_dir train/log_v2 --num_point 512 --max_epoch 11 --batch_size 32 --decay_step 800000 --decay_rate 0.5  --data_path /root/frustum-pointnets_RSC/dataset/  --pkl_output_path /root/frustum-pointnets_RSC_RADAR_fil_PC_batch_para/dataset/RSC/seg_cls/
  
  Same output schemas is used here as Frustum-PointNet.
  _Radar-PointNEt-Para w/o  classification_
  
  ![teaser](https://github.com/ben0110/Radar-PointNet/blob/master/pictures/RADAR-pointnet-para_cls.jpg)
  
  
  Train the segmenetation network with the classification network with the cmd:
  
    python train/train_seg_cls.py --gpu 1 --model frustum_pointnets_seg_cls_v2  --log_dir train/log_v2 --num_point 25000 --max_epoch 201 --batch_size 4 --decay_step 800000 --decay_rate 0.5  --data_path /root/frustum-pointnets_RSC/dataset/ --pkl_output_path /root/frustum-pointnets_RSC_RADAR_fil_PC_batch_para/dataset/RSC/seg_cls/
  
  This will output the train, val, results in the form of pickle file which contains the frame id,the detected object pc present in each proposals and its coresponding classsfication score. The quantitaive results (average segmentation accuracy, average classification network) are stored in the corresponding logfile.
  
  Training the bbox network from the corresponding semgentation output is done through the cmd:
  
     python train/train_bbox_cls.py --gpu 1 --model frustum_pointnets_bbox_v2  --log_dir train/log_v2 --num_point 512 --max_epoch 11 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --data_path /root/frustum-pointnets_RSC/dataset/  --pkl_output_path /root/frustum-pointnets_RSC_RADAR_fil_PC_batch_para/dataset/RSC/seg_cls/
  
  
  Same output schemas is used here as Frustum-PointNet.
    
  
  
  
    

 
 
 
 
 
 
 
