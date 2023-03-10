PYTHON="/home2/ahasssan/anaconda3/envs/myenv/bin/python"

lambda=0.9

$PYTHON trainer.py \
    --epochs 40 \
    --architecture object-detection \
    --process test \
    --model Yolov2 \
    --TET True \
    --lr 5e-4 \
    --lamb ${lambda} \
    --dataset  Gen1 \
    --batch-size 32 \
    --frequency 500 \
    --lvth False \
    --save_images "./test/images/Train_histogram_detect_bounding boxes.png" \
    --save_path "./train/paths/SGP_Yolov2_SNN_hist_TET_0.9_Object_detection_model_40epoch_Learnable_Vth.pth"