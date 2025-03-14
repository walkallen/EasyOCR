

## 推理命令

```bash
# 启用 gpu 推理
python -m easyocr.cli    -l ch_sim  -f ./examples/s4.png --detail=1   --verbose True --gpu

# 关闭 gpu 推理
python -m easyocr.cli    -l ch_sim  -f ./examples/s4.png --detail=1   --verbose True 
```

## 训练命令

```bash
cd trainer
python runtrain.py
```