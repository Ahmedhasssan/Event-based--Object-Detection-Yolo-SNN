# Object-detection-with-LT-SNN-YoloV2
This repo presents the LT_SNN based custom YoloV2 architecture to perform object detection on DVS object detection datasets.

Dependencies:

      -Pytorch 1.9.0+Cu111
      -Device: GPU
  
Dataset:

      -Prophesee Gen1 and Gen4 Automotive data
      -Download dataset from "https://docs.prophesee.ai/stable/datasets.html"
      -Dataset folder: Change root path in trainer.py
      -We have borrowed prophesee dataloader from "https://github.com/uzh-rpg/rpg_asynet"
      
Training:
      
      -Bash run_gen1.sh
            -Provide checkpoints path to save the model
            -Provide the path to save training performance 

Inference:

      -Bash run_gen1_inference.sh
            -Provide saved model path
            -Provide the path to save inference images
            
<img
  src="./train/images/Train_histogram_detect_bounding boxes.png"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 200px">
  
            
