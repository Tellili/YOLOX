TRAIN_CMD = python tools/train.py \
	-f exps/example/custom/yolox_m.py \
	-b 8 --fp16 \
	-o -c /data/haythem/openml/YOLOX/weights/yolox_m.pth

EXPORT_ONNX_CMD = python3 tools/export_onnx.py \
	--output-name /data/haythem/openml/YOLOX/YOLOX_outputs/yolox_sv1/yolox_s.onnx \
	-f /data/haythem/openml/YOLOX/exps/example/custom/yolox_s.py \
	-c /data/haythem/openml/YOLOX/YOLOX_outputs/yolox_sv1/best_ckpt.pth

INFERENCE_CMD = python3 demo/ONNXRuntime/onnx_inference.py \
	-m /data/haythem/openml/YOLOX/YOLOX_outputs/yolox_sv1/yolox_s.onnx \
	-i "/data/haythem/openml/YOLOX/datasets/3k_dataset/data/0a1bdc02818d42adbe6e7e16e6e75bcd.JPG" \
	-o /data/haythem/openml/YOLOX/YOLOX_outputs/inf_result \
	-s 0.3 --input_shape 640,640

INFERENCE_TIME_CMD = python3 demo/ONNXRuntime/onnx_inference_time.py \
	-m /data/haythem/openml/YOLOX/YOLOX_outputs/yolox_sv1/yolox_s.onnx \
	-i "/data/haythem/openml/YOLOX/datasets/3k_dataset/data/0a1bdc02818d42adbe6e7e16e6e75bcd.JPG" \
	-o /data/haythem/openml/YOLOX/YOLOX_outputs/inf_result \
	-s 0.3 --input_shape 640,640

train:
	$(TRAIN_CMD)

export_onnx:
	$(EXPORT_ONNX_CMD)

inference:
	$(INFERENCE_CMD)

inference_time:
	$(INFERENCE_TIME_CMD)