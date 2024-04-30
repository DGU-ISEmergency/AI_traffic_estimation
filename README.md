# Official YOLOv7
train yolov7x model

## Training

``` shell
python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --workers 8 --device 0,1 --sync-bn --batch-size 8 --data data/coco.yaml --img 1080 1920 --cfg cfg/training/yolov7x.yaml --weights '' --name yolov7x --hyp data/hyp.scratch.p5.yaml --epochs 80
```



## Tracking 추가
config default 값 이미 설정해놓음

`runs/train/yolov7x2/weights/best.pt` 의 weight 값을 사용하여 tracking을 진행함

``` shell
python detect_and_track.py
```