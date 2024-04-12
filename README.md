# Official YOLOv7
train yolov7x model

## Training

``` shell
python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --workers 8 --device 0,1 --sync-bn --batch-size 8 --data data/coco.yaml --img 1080 1920 --cfg cfg/training/yolov7x.yaml --weights '' --name yolov7x --hyp data/hyp.scratch.p5.yaml --epochs 80
```
