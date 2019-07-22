# Object Detection
## SSD -> ssd_mobilenet_v2_coco(tensorflow)
```bash
python3 ./downloader.py --name ssd_mobilenet_v2_coco -o /home/intel/user/model_forSummmerCamp/object_detection
#fp32
/opt/openvino_toolkit/dldt/model-optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/openvino_toolkit/dldt/model-optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --data_type FP32 --output_dir ./output/fp32/
#fp16
/opt/openvino_toolkit/dldt/model-optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/openvino_toolkit/dldt/model-optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --data_type FP16 --output_dir ./output/fp16/
```
refer to:
1. [https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html)
2. [https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html)
3. [https://github.com/opencv/dldt/issues/75](https://github.com/opencv/dldt/issues/75)
4. [https://gist.github.com/aallan/fbdf008cffd1e08a619ad11a02b74fa8](https://gist.github.com/aallan/fbdf008cffd1e08a619ad11a02b74fa8)
## Yolov2-voc

# People Detection

# Segmentation -> 
```bash
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz
tar -zxvf faster_rcnn_resnet50_coco_2018_01_28.tar.gz
#fp32
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --data_type FP32 --output_dir ./output/FP32
#fp16
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --data_type FP16 --output_dir ./output/FP16
```
refer to:


# People Reid

# face Reid
