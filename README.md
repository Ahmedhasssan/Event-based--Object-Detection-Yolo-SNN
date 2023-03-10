# Object-detection-with-LT-SNN-YoloV2
This repo presents the LT_SNN based custom YoloV2 architecture to perform object detection on DVS object detection datasets.

Dependencies:

      -Pytorch 1.9.0+Cu111
      -Device: GPU
  
Dataset:

      -Prophesee Gen1 and Gen4 Automotive data
      -Download dataset from "https://docs.prophesee.ai/stable/datasets.html"
      -Dataset folder: Change root path in trainer.py
      
Training:
      
      -Bash run_gen1.sh
      
            -Provide checkpoints path to save the model
            -Provide the path to save training performance 

Inference:

      -Bash run_gen1_inference.sh
            
            -Provide saved model path
            -Provide the path to save inference images
            
