

## 推理命令

```bash
# 启用 gpu 推理
python -m easyocr.cli    -l ch_sim  -f ./examples/s4.png --detail=1   --verbose True --gpu

# 关闭 gpu 推理
python -m easyocr.cli    -l ch_sim  -f ./examples/s4.png --detail=1   --verbose True 


time for file in ./input/* ;do  python -m easyocr.cli    -l ch_sim en  -f "$file" --detail=0  --gpu;done


python -m easyocr.cli    -l ch_sim en  -f 'examples/input/生产日期 2024 December 1204_3.jpg' --detail=0


# 如果要使用 openvino 模型进行推理
python -m easyocr.cli    -l ch_sim en  -f 'examples/input/生产日期 2024 December 1204_3.jpg' --detail=0 --openvino

```

## 训练命令

```bash
cd trainer
python runtrain.py

```


将中文训练图片放在 trainer/trdata/  和 trainer/vadata/ 下面，从头开始训练少数几个中文图片

```bash
cd trainer
python runtrain_mixed.py

```



## 更新日志




--------------------------------------------------------------------------------------------------------

为检测模型增加 openvino 模式，在推理命令后增加 --openvino 选项即可


--------------------------------------------------------------------------------------------------------