# -*- coding: utf-8 -*-
from flask import *
from fileinput import filename

import urllib.request

import tuna

import threading, queue
import zipfile
import requests
from pathlib import Path

import sys, json, os, time, logging, random, shutil, tempfile, subprocess, re, platform, io
import base64
import numpy as np
np.bool = np.bool_
np.int = np.int_
np.float = np.float32
import cv2
#---- helper ----#
from utils.message_announcer import MessageAnnouncer
import utils.helper as helper
sys.path.append(".")
#---- train ----#
from train_object_detection import train_object_detection
from train_image_classification import train_image_classification
from train_voice_classification import train_voice_classification
from train_object_detection_yolo11 import train_object_detection_yolo11
#---- converter ----#
from convert import torch_to_onnx, onnx_to_ncnn, gen_input
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import *
from utils.modules import replace_relu6_with_relu

app = Flask(__name__)

#==================================== Define Variables ====================================#
UNAME = platform.uname()
BACKEND = ""
DEVICE = ""
if UNAME.system == "Windows":
    BACKEND = "EDGE"
    DEVICE = "WINDOWS"
elif 'COLAB_GPU' in os.environ:
    BACKEND = "COLAB"
    DEVICE = "COLAB"
elif os.path.exists("/proc/device-tree/model"):
    with open("/proc/device-tree/model", "r") as f:
        model = f.read().strip()
        if "Jetson Nano" in model:
            DEVICE = "JETSON"
            BACKEND = "EDGE"
        elif "Raspberry Pi" in model:
            DEVICE = "RPI"
            BACKEND = "EDGE"
        elif "Nano" in model:
            DEVICE = "NANO"
            BACKEND = "EDGE"
else:
    # Generic Linux / WSL / Docker — e.g. desktop with CUDA GPU
    BACKEND = "EDGE"
    DEVICE = "WSL" if "microsoft" in UNAME.release.lower() else "LINUX"

PROJECT_PATH = "./projects" if BACKEND == "COLAB" else "./projects"
PROJECT_FILENAME = "project.json"
PROJECT_ZIP = "project.zip"
OUTPUT_FOLDER = "output"
TEMP_FOLDER = "temp"

STAGE = 0 #0 none, 1 = prepare dataset, 2 = training, 3 = trained, 4 = converting, 5 converted

reporter = MessageAnnouncer()
train_task = None
convert_task = None

#==================================== Server Configuration ====================================#
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
logging.basicConfig(level=logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

@app.route('/')
def index():
    return "Hello World"

@app.route('/listen', methods=['GET'])
def listen():
    def stream():
        messages = reporter.listen()  # returns a queue.Queue
        while True:
            msg = messages.get()  # blocks until a new message arrives
            yield msg

    return Response(stream(), mimetype='text/event-stream')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':        
        f = request.files['project']
        project_id = request.form['project_id']
        project_path = os.path.join(PROJECT_PATH, project_id)
        helper.recreate_folder(project_path)
        f.save(os.path.join(project_path, PROJECT_ZIP))
        
        return jsonify({'result': 'success'})

@app.route("/train", methods=["POST"])
def start_training():
    global train_task, reporter
    print("start training process")
    data = request.get_json()
    project_id = data["project"]
    train_task = threading.Thread(target=training_task, args=(project_id,reporter,))
    train_task.start()
    return jsonify({"result" : "OK"})

@app.route("/convert", methods=["GET"])
def download_file():
    global convert_task, reporter
    # convert project
    project_id = request.args.get("project_id")
    if not project_id:
        return "Fail"
    #convert_task = threading.Thread(target=convert_model, args=(project_id,reporter,))
    #convert_task.start()
    convert_model(project_id, reporter)
    return jsonify({"result" : "OK"})
    
@app.route("/download_model", methods=["GET"])
def handle_download_model():
    print("download model file")
    project_id = request.args.get("project_id")
    model_export = os.path.join(PROJECT_PATH,project_id,"model.zip")
    return send_file(model_export, as_attachment=True)


@app.route('/ping', methods=["GET","POST"])
def on_ping():
    return jsonify({"result":"pong", "device":DEVICE, "backend":BACKEND, "stage":STAGE})

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Methods']='*'
    response.headers['Access-Control-Allow-Origin']='*'
    response.headers['Vary']='Origin'
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Access-Control-Allow-Headers, X-Requested-With"
    return response

@app.route('/projects/<path:path>')
def send_report(path):
    return send_from_directory('projects', path)

def convert_model(project_id, q):
    global STAGE

    STAGE = 4
    project_path = os.path.join(PROJECT_PATH, project_id)
    project = helper.read_json_file(os.path.join(project_path, PROJECT_FILENAME))
    
    best_file = None
    modelType = "code" if "code" in project["trainConfig"] else project["trainConfig"]["modelType"]

    if modelType.startswith("mobilenet") or modelType in ("resnet18", "code", "voice-cnn"):
        best_file = os.path.join(project_path, "output", "best_acc.pth")

    elif modelType == "slim_yolo_v2":
        best_file = os.path.join(project_path, "output", "best_map.pth")
        
    elif modelType in ("yolo11n", "yolo11s"):
        run_name = f"{modelType}_run"
        best_file = os.path.join(project_path, "output", run_name, "weights", "best.pt")
        if not os.path.exists(best_file):
            alt_best_file = os.path.join("runs", "detect", project_path, "output", run_name, "weights", "best.pt")
            if os.path.exists(alt_best_file):
                best_file = alt_best_file

    if best_file == None or not os.path.exists(best_file):
        return q.announce({"time":time.time(), "event": "error", "msg" : "No best_map.pth file"})
    
    device = torch.device("cpu")
    
    q.announce({"time":time.time(), "event": "convert_model_init", "msg" : "Start converting model"})

    #load project
    model_label = [l["label"] for l in project["labels"]]
    model_label.sort()
    num_classes = len(model_label)
    if modelType == "slim_yolo_v2":
        input_size = [224, 224]        
        print("label:", model_label)
        if modelType == "slim_yolo_v2":
            from models.slim_yolo_v2 import SlimYOLOv2
            anchor_size = ANCHOR_SIZE            
            detect_threshold = float(project["trainConfig"]["objectThreshold"])
            iou_threshold = float(project["trainConfig"]["iouThreshold"])
            net = SlimYOLOv2(device, input_size=input_size, num_classes=num_classes, conf_thresh=detect_threshold, nms_thresh=iou_threshold, anchor_size=anchor_size)

        # strict=False tolerates QAT buffers (act_scale/_qat_steps) present when
        # the checkpoint was trained under KBMAI_USE_QAT=1. Weights still load.
        net.load_state_dict(torch.load(best_file, map_location=device), strict=False)
        net.to(device).eval()

    elif modelType.startswith("mobilenet"):
        input_size = [224, 224]
        model_label = [ l["label"] for l in project["labels"]]
        model_label.sort()
        from torchvision.models import mobilenet_v2
        net = mobilenet_v2(pretrained=False)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
        # Must match the ReLU6→ReLU swap applied during training, otherwise the
        # state_dict layer names still match but ONNX export emits ReLU6 ops.
        replace_relu6_with_relu(net)
        net.load_state_dict(torch.load(best_file, map_location=device))
        net.to(device).eval()

    elif modelType == "resnet18":
        input_size = [224, 224]
        model_label = [ l["label"] for l in project["labels"]]
        model_label.sort()
        from torchvision import models
        net = models.resnet18(pretrained=False)
        net.fc = nn.Linear(net.fc.in_features , num_classes)
        net.load_state_dict(torch.load(best_file, map_location=device))
        net.to(device).eval()

    elif modelType == "voice-cnn":
        _H = int(os.environ.get("KBMAI_VOICE_INPUT_H", "40"))
        _W = int(os.environ.get("KBMAI_VOICE_INPUT_W", "147"))
        input_size = [_H, _W]
        model_label = [ l["label"] for l in project["labels"]]
        model_label.sort()
        from models.voice_cnn import VoiceCNN
        net = VoiceCNN(num_classes=num_classes)
        net.load_state_dict(torch.load(best_file, map_location=device))
        net.to(device).eval()

    elif modelType == "code" and project["projectType"] == "VOICE_CLASSIFICATION":
        input_size = [13, 147]
        model_label = [ l["label"] for l in project["labels"]]
        model_label.sort()
        from models.voice_cnn import VoiceCnn
        net = VoiceCnn(device, project["trainConfig"]["code"], input_size=input_size, num_classes=num_classes, trainable=False)
        net.load_state_dict(torch.load(best_file, map_location=device))
        net.to(device).eval()

    elif modelType in ("yolo11n", "yolo11s"):
        input_size = [224, 320]
        net = None


    print('Finished loading model!')

    # Voice-cnn on V831 (kidbright-mai) always takes the numpy fp32 CPU path.
    # Every attempt at INT8 (spnntools per-tensor, modern ncnn per-channel, QAT,
    # mel-spec, arch variants) collapses the tiny classifier head. Export the
    # trained weights as a single .npz that voice_cpu_infer.py on-board can load
    # and forward through numpy matmul at ~50-200 ms/clip.
    board_id_early = project.get("currentBoard", {}).get("id", "")
    if modelType == "voice-cnn" and board_id_early == "kidbright-mai":
        npz_out = os.path.join(project_path, "output", "model_cpu.npz")
        sd_np = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        np.savez(npz_out, labels=np.array(model_label), **sd_np)
        q.announce({"time": time.time(), "event": "initial",
                    "msg": f"voice: saved fp32 numpy weights to {os.path.basename(npz_out)} "
                           f"({os.path.getsize(npz_out)} B)"})
        STAGE = 5
        q.announce({"time": time.time(), "event": "convert_model_end",
                    "msg": "Model converted successfully (voice CPU path)"})
        return

    if modelType not in ("yolo11n", "yolo11s"):
        # convert to onnx and ncnn
        from torchsummary import summary
        summary(net.to("cpu"), input_size=(3, input_size[0], input_size[1]), device="cpu")
    
        # convert model    
        net.no_post_process = True
        onnx_out= os.path.join(project_path, "output", "model.onnx")
        ncnn_out_param = os.path.join(project_path, "output", "model.param")
        ncnn_out_bin = os.path.join(project_path, "output", "model.bin")
        input_shape = (3, input_size[0], input_size[1])
        
        opencv_root = os.environ.get("KBMAI_OPENCV_ROOT", "/root/opencv-3.4.13")
        extra_lib = f'{opencv_root}/lib_extra'
        os.environ['PKG_CONFIG_PATH'] = os.environ.get('PKG_CONFIG_PATH', '') + f':{opencv_root}/lib/pkgconfig'
        ld = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = f'{ld}:{opencv_root}/lib:{extra_lib}'
        os.environ['PATH'] = os.environ.get('PATH', '') + f':{opencv_root}/bin'
    
        with torch.no_grad():
            q.announce({"time":time.time(), "event": "initial", "msg" : "Start converting model to onnx"})
            torch_to_onnx(net.to("cpu"), input_shape, onnx_out, device="cpu")
        net.no_post_process = False
    elif modelType in ("yolo11n", "yolo11s"):
        # Export YOLO11 to ONNX using Ultralytics
        from ultralytics import YOLO
        yolo_model = YOLO(best_file)
        q.announce({"time":time.time(), "event": "initial", "msg" : "Start converting YOLO model to onnx"})
        
        onnx_out = os.path.join(project_path, "output", "model.onnx")
        
        # YOLO export returns the exported onnx object path
        exported_path = yolo_model.export(format="onnx", imgsz=[224, 320]) #, opset=11)
        
        if exported_path and os.path.exists(exported_path):
            shutil.move(exported_path, onnx_out)
        else:
            q.announce({"time":time.time(), "event": "error", "msg" : "YOLO ONNX export failed"})
            return

    board_id = project.get("currentBoard", {}).get("id", "")
    if modelType == "slim_yolo_v2":
        # Allow overriding the calibration folder for slim_yolo_v2 so tests can
        # substitute an augmented set (wider pixel/brightness coverage → better
        # INT8 range estimates than the tiny raw JPEGImages dir).
        override = os.environ.get("KBMAI_SLIM_CALIB_DIR")
        if override and os.path.isdir(override):
            images_path = override
            q.announce({"time": time.time(), "event": "initial",
                        "msg": f"slim_yolo_v2 calibration using override dir: {override}"})
        else:
            images_path = os.path.join(project_path, "dataset", "JPEGImages")
    elif modelType == "voice-cnn" or (modelType == "code" and project.get("projectType") == "VOICE_CLASSIFICATION"):
        # Voice MFCC spectrograms have very different activation statistics from
        # natural images, so calibrating against data/test_images (face photos)
        # produces int8 thresholds that don't cover the real inference range and
        # the quantized model collapses to a constant class. Build a mixed MFCC
        # calibration set from the project's own training data.
        mfcc_cal_dir = os.path.join(project_path, "output", "cal_mfcc")
        if os.path.exists(mfcc_cal_dir):
            shutil.rmtree(mfcc_cal_dir)
        os.makedirs(mfcc_cal_dir, exist_ok=True)
        mfcc_root = os.path.join(project_path, "dataset", "mfcc")
        # training_task() moves samples into train/<label>/ before saving the pth;
        # fall back to the flat <label>/ layout if we're converting without a retrain.
        src_root = os.path.join(mfcc_root, "train")
        if not os.path.isdir(src_root):
            src_root = mfcc_root
        # Pre-resize MFCC PNGs to (input_size[1], input_size[0]) = (W, H) so
        # spnntools calibrates on the same tensor shape the network will see
        # (grayscale replicated to 3 channels, matching training's Grayscale(3) +
        # ImageFolder RGB loader).
        from PIL import Image as _PImg
        count = 0
        for cls in sorted(os.listdir(src_root)):
            cls_dir = os.path.join(src_root, cls)
            if not os.path.isdir(cls_dir) or cls in ("train", "valid"):
                continue
            for fn in sorted(os.listdir(cls_dir))[:40]:
                img = _PImg.open(os.path.join(cls_dir, fn)).convert("L")
                img = img.resize((input_size[1], input_size[0]), _PImg.BILINEAR)
                img = img.convert("RGB")
                img.save(os.path.join(mfcc_cal_dir, f"{cls}_{fn}"))
                count += 1
        images_path = mfcc_cal_dir
        q.announce({"time": time.time(), "event": "initial", "msg": f"voice calibration set: {count} MFCC images"})
        print(f"[calibrate] voice uses {count} MFCC images from {src_root}")
    else:
        images_path = os.path.join("data", "test_images")

    if board_id == "kidbright-mai-plus" and (modelType.startswith("mobilenet") or modelType in ("resnet18", "voice-cnn")):
        q.announce({"time":time.time(), "event": "initial", "msg" : "Start converting onnx to cvimodel"})
        mlir_out = os.path.join(project_path, "output", "mobilenet.mlir")
        npz_out = os.path.join(project_path, "output", "mobilenet_top_outputs.npz")
        cali_table_out = os.path.join(project_path, "output", "mobilenet_cali_table")
        cvimodel_out = os.path.join(project_path, "output", "model.cvimodel")
        
        test_img = os.path.join("data", "test_images2", "cat.jpg")

        cmd1 = f"conda run -n kbmai model_transform.py --model_name mobilenet --model_def {onnx_out} --input_shapes [[1,3,{input_size[0]},{input_size[1]}]] --mean 123.675,116.28,103.53 --scale 0.0171,0.0175,0.0174 --keep_aspect_ratio --pixel_format rgb --channel_format nchw --test_input {test_img} --test_result {npz_out} --tolerance 0.99,0.99 --mlir {mlir_out}"
        cmd2 = f"conda run -n kbmai run_calibration.py {mlir_out} --dataset {images_path} --input_num 24 --processor cv181x -o {cali_table_out}"
        cmd3 = f"conda run -n kbmai model_deploy.py --mlir {mlir_out} --quantize INT8 --quant_input --calibration_table {cali_table_out} --chip cv181x --processor cv181x --test_input {npz_out} --test_reference {npz_out} --tolerance 0.9,0.6 --model {cvimodel_out}"
        
        os.system(cmd1)
        os.system(cmd2)
        os.system(cmd3)

        if os.path.exists(cvimodel_out):
            mud_out = os.path.join(project_path, "output", "model.mud")
            labels_str = ", ".join(model_label)
            mud_content = (
                "[basic]\n"
                "type = cvimodel\n"
                "model = model.cvimodel\n\n"
                "[extra]\n"
                "model_type = classifier\n"
                "input_type = rgb\n"
                "mean = 123.675, 116.28, 103.53\n"
                "scale = 0.017124753831663668, 0.01750700280112045, 0.017429193899782137\n"
                f"labels = {labels_str}\n"
            )
            with open(mud_out, "w") as f:
                f.write(mud_content)
            q.announce({"time":time.time(), "event": "initial", "msg" : "Created model.mud"})
        else:
            q.announce({"time":time.time(), "event": "error", "msg" : "Failed to generate cvimodel"})

    elif board_id == "kidbright-mai-plus" and modelType in ("yolo11n", "yolo11s"):
        q.announce({"time":time.time(), "event": "initial", "msg" : f"Start converting ONNX to cvimodel for {modelType}"})
        mlir_out = os.path.join(project_path, "output", "model.mlir")
        npz_out = os.path.join(project_path, "output", f"{modelType}_top_outputs.npz")
        npz_in_out = f"{modelType}_in_f32.npz"
        cali_table_out = os.path.join(project_path, "output", f"{modelType}_cali_table")
        cvimodel_out = os.path.join(project_path, "output", "model.cvimodel")

        test_img_transform = os.path.join("data", "test_images2", "cat.jpg")

        output_names = "/model.23/dfl/conv/Conv_output_0,/model.23/Sigmoid_output_0"

        deploy_tolerance = {"yolo11n": "0.9,0.6", "yolo11s": "0.85,0.5"}[modelType]

        cmd1_list = [
            f"conda run -n kbmai model_transform.py",
            f"--model_name {modelType}",
            f"--model_def {onnx_out}",
            f"--input_shapes [[1,3,224,320]]",
            f"--mean \"0,0,0\"",
            f"--output_names \"{output_names}\"",
            f"--scale \"0.00392156862745098,0.00392156862745098,0.00392156862745098\"",
            f"--keep_aspect_ratio",
            f"--pixel_format rgb",
            f"--channel_format nchw",
            f"--test_input {test_img_transform}",
            f"--test_result {npz_out}",
            f"--tolerance 0.99,0.99",
            f"--mlir {mlir_out}"
        ]
        cmd1 = " ".join(cmd1_list)

        dataset_path_cali = os.path.join("data", "test_images")

        cmd2_list = [
            f"conda run -n kbmai run_calibration.py",
            f"{mlir_out}",
            f"--dataset {dataset_path_cali}",
            f"--input_num 24",
            f"-o {cali_table_out}"
        ]
        cmd2 = " ".join(cmd2_list)

        cmd3_list = [
            f"conda run -n kbmai model_deploy.py",
            f"--mlir {mlir_out}",
            f"--quantize INT8",
            f"--calibration_table {cali_table_out}",
            f"--processor cv181x",
            f"--test_input {npz_in_out}",
            f"--test_reference {npz_out}",
            f"--tolerance {deploy_tolerance}",
            f"--model {cvimodel_out}"
        ]
        cmd3 = " ".join(cmd3_list)

        print("Running CMD1:", cmd1)
        os.system(cmd1)
        print("Running CMD2:", cmd2)
        os.system(cmd2)
        print("Running CMD3:", cmd3)
        os.system(cmd3)

        if os.path.exists(cvimodel_out):
            mud_out = os.path.join(project_path, "output", "model.mud")
            labels_str = ", ".join(model_label)
            mud_content = (
                "[basic]\n"
                "type = cvimodel\n"
                "model = model.cvimodel\n\n"
                "[extra]\n"
                "model_type = yolo11\n"
                "input_type = rgb\n"
                "mean = 0, 0, 0\n"
                "scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098\n"
                f"labels = {labels_str}\n"
            )
            with open(mud_out, "w") as f:
                f.write(mud_content)
            q.announce({"time":time.time(), "event": "initial", "msg" : "Created model.mud"})
        else:
            q.announce({"time":time.time(), "event": "error", "msg" : "Failed to generate cvimodel"})

    else:
        with torch.no_grad():
            q.announce({"time":time.time(), "event": "initial", "msg" : "Start converting onnx to ncnn"})
            onnx_to_ncnn(input_shape, onnx=onnx_out, ncnn_param=ncnn_out_param, ncnn_bin=ncnn_out_bin)
            print("convert end, ctrl-c to exit")

        output_model_optimize_bin_path = os.path.join(project_path, "output", "model_opt.bin")
        output_model_optimize_param_path = os.path.join(project_path, "output", "model_opt.param")
        q.announce({"time":time.time(), "event": "initial", "msg" : "Start optimizing model"})
        cmd = f"tools/spnntools optimize {ncnn_out_param} {ncnn_out_bin} {output_model_optimize_param_path} {output_model_optimize_bin_path}"
        os.system(cmd)

        output_model_calibrate_table = os.path.join(project_path, "output", "model_opt.table")
        q.announce({"time":time.time(), "event": "initial", "msg" : "Start calibrating model"})
        # V831 AWNN expects normalized input within ~[-1, +1]. Training + calibration +
        # inference all use (x-127.5)/128 so the network sees the same range end-to-end.
        # See train_image_classification.py for rationale.
        calib_mean = "127.5,127.5,127.5"
        calib_norm = "0.0078125,0.0078125,0.0078125"
        # spnntools --size is W,H (width,height) but input_size in this file is (H,W)
        # (matches torch.Tensor (C,H,W) convention used in torch_to_onnx). Pass W first
        # so non-square inputs (voice 13x147, yolo11n 224x320) calibrate on correctly
        # oriented images.
        cmd2 = "tools/spnntools calibrate -p=\""+output_model_optimize_param_path+"\" -b=\""+output_model_optimize_bin_path+"\" -i=\""+images_path+"\" -o=\""+output_model_calibrate_table+"\" --m=\""+calib_mean+"\" --n=\""+calib_norm+"\" --size=\""+str(input_size[1])+","+str(input_size[0])+"\" -c -t=4"
        os.system(cmd2)

        output_model_quantize_bin_path = os.path.join(project_path, "output", "model_int8.bin")
        output_model_quantize_param_path = os.path.join(project_path, "output", "model_int8.param")
        q.announce({"time":time.time(), "event": "initial", "msg" : "Start quantizing model"})
        cmd3 = "tools/spnntools quantize "+output_model_optimize_param_path+" "+output_model_optimize_bin_path+" "+output_model_quantize_param_path+" "+output_model_quantize_bin_path+" "+output_model_calibrate_table
        os.system(cmd3)
        
    STAGE = 5
    q.announce({"time":time.time(), "event": "convert_model_end", "msg" : "Model converted successfully"})
    

def training_task(project_id, q):
    global STAGE, current_model
    try:
        # 1 ========== prepare project ========= #
        STAGE = 1
        # for i in range(50):
        #     q.announce({"time":time.time(), "event": "initial", "msg" : "Start training step 1 ... prepare dataset"})
        #     time.sleep(1)
        q.announce({"time":time.time(), "event": "initial", "msg" : "Start training step 1 ... prepare dataset"})
        
        # unzip project (skip if already extracted — CLI / re-run case)
        project_zip = os.path.join(PROJECT_PATH, project_id, PROJECT_ZIP)
        project_folder = os.path.join(PROJECT_PATH, project_id)
        if os.path.exists(project_zip):
            with zipfile.ZipFile(project_zip, 'r') as zip_ref:
                zip_ref.extractall(project_folder)
            os.remove(project_zip)
        # read project file
        project = helper.read_json_file(os.path.join(PROJECT_PATH, project_id, PROJECT_FILENAME))        
        q.announce({"time":time.time(), "event": "initial", "msg" : "target project id : "+project_id})
        # 2 ========== prepare dataset ========= #
        STAGE = 2
        # execute script "!python train.py -d custom --cuda -v slim_yolo_v2 -hr -ms"
        q.announce({"time":time.time(), "event": "initial", "msg" : "Start training step 2 ... training"})
        
        output_path = os.path.join(project_folder, "output")
        dataset_path = os.path.join(project_folder, "datasets")
        #{'validateMatrix': 'validation-accuracy', 'saveMethod': 'Best value after n epoch', 'modelType': 'Resnet18', 'weights': 'resnet18', 'inputWidth': 320, 'inputHeight': 240, 'train_split': 80, 'epochs': 100, 'batch_size': 32, 'learning_rate': 0.001}
        # check if project has trainConfig and it valid        

        
        # 3 ========== training ========= #
        # label format in json lables : [ {label: "label1"}, {label: "label2"}]
        model_label = [l["label"] for l in project["labels"]]
        model_label.sort()
        modelType = "code" if "code" in project["trainConfig"] else project["trainConfig"]["modelType"]
        if project["projectType"] == "VOICE_CLASSIFICATION":
            # Regenerate dataset/mfcc/ as 40-bin log-mel from the WAVs so training
            # features match the on-board CPU inference pipeline. Auto-detect the
            # input W (frames) from the first WAV's sample count so projects with
            # any recording duration (1 s, 3 s, ...) work without the user
            # touching env vars.
            sound_dir = os.path.join(project_folder, "dataset", "sound")
            mfcc_dir = os.path.join(project_folder, "dataset", "mfcc")
            if os.path.isdir(sound_dir):
                import regen_melspec as _rms
                import wave as _wv
                detected_W = None
                for cls in sorted(os.listdir(sound_dir)):
                    cdir = os.path.join(sound_dir, cls)
                    if not os.path.isdir(cdir):
                        continue
                    wavs = [f for f in sorted(os.listdir(cdir)) if f.endswith(".wav")]
                    if not wavs:
                        continue
                    with _wv.open(os.path.join(cdir, wavs[0]), "rb") as _wf:
                        _nsamples = _wf.getnframes()
                    FL = int(0.040 * 44100); FS = int(0.040 * 44100 / 2)
                    detected_W = int((_nsamples - FL) / FS)
                    break
                if detected_W is None:
                    detected_W = 147  # reasonable default for 3 s audio
                q.announce({"time": time.time(), "event": "initial",
                            "msg": f"voice: auto-detected input_W={detected_W} from wav"})
                # Replace any IDE-generated MFCC PNGs with fresh log-mel from wav
                if os.path.isdir(mfcc_dir):
                    shutil.rmtree(mfcc_dir)
                # Call regen_melspec.main() by argv injection (keeps its signature simple)
                _saved_argv = sys.argv
                sys.argv = ["regen_melspec.py", sound_dir, mfcc_dir]
                _rms.main()
                sys.argv = _saved_argv
                os.environ["KBMAI_VOICE_INPUT_H"] = "40"
                os.environ["KBMAI_VOICE_INPUT_W"] = str(detected_W)
            res = train_voice_classification(project, output_path, project_folder,q,
                cuda= True if torch.cuda.is_available() else False, 
                learning_rate=project["trainConfig"]["learning_rate"],  
                batch_size=project["trainConfig"]["batch_size"],
                start_epoch=0, 
                epoch=project["trainConfig"]["epochs"],
                train_split=project["trainConfig"]["train_split"], 
                model_type=modelType, 
                model_weight=None,
                validate_matrix='val_acc',
                save_method=project["trainConfig"]["saveMethod"],
                step_lr=(150, 200),
                labels=model_label,
                weight_decay=5e-4,
                warm_up_epoch=6
            )
            if res:
                STAGE = 3
        elif modelType == "slim_yolo_v2":
            res = train_object_detection(project, output_path, project_folder,q,
                high_resolution=True, 
                multi_scale=True, 
                cuda= True if torch.cuda.is_available() else False, 
                learning_rate=project["trainConfig"]["learning_rate"], 
                batch_size=project["trainConfig"]["batch_size"],
                start_epoch=0, 
                epoch=project["trainConfig"]["epochs"],
                train_split=project["trainConfig"]["train_split"],
                model_type=modelType,
                model_weight=None,
                validate_matrix=project["trainConfig"]["validateMatrix"],
                save_method=project["trainConfig"]["saveMethod"],
                step_lr=(150, 200),
                labels=model_label,
                momentum=0.9,
                weight_decay=5e-4,
                warm_up_epoch=6
            )
            if res:
                STAGE = 3
        
        elif modelType in ("yolo11n", "yolo11s"):
            res = train_object_detection_yolo11(project, output_path, project_folder,q,
                high_resolution=True, 
                multi_scale=True, 
                cuda= True if torch.cuda.is_available() else False, 
                learning_rate=project["trainConfig"]["learning_rate"], 
                batch_size=project["trainConfig"]["batch_size"],
                start_epoch=0, 
                epoch=project["trainConfig"]["epochs"],
                train_split=project["trainConfig"]["train_split"],
                model_type=modelType,
                model_weight=None,
                validate_matrix=project["trainConfig"]["validateMatrix"],
                save_method=project["trainConfig"]["saveMethod"],
                step_lr=(150, 200),
                labels=model_label,
                momentum=0.9,
                weight_decay=5e-4,
                warm_up_epoch=6
            )
            if res:
                STAGE = 3
        
        #check if start with resnet18
        elif modelType.startswith("resnet"):
            res = train_image_classification(project, output_path, project_folder,q,
                cuda= True if torch.cuda.is_available() else False, 
                learning_rate=project["trainConfig"]["learning_rate"],  
                batch_size=project["trainConfig"]["batch_size"],
                start_epoch=0, 
                epoch=project["trainConfig"]["epochs"],
                train_split=project["trainConfig"]["train_split"], 
                model_type=modelType, 
                model_weight=None,
                validate_matrix='val_acc',
                save_method=project["trainConfig"]["saveMethod"],
                step_lr=(150, 200),
                labels=model_label,
                weight_decay=5e-4,
                warm_up_epoch=6
            )
            if res:
                STAGE = 3

        #check if start with mobile net
        elif modelType.startswith("mobilenet"):
            res = train_image_classification(project, output_path, project_folder,q,
                cuda= True if torch.cuda.is_available() else False, 
                learning_rate=project["trainConfig"]["learning_rate"],  
                batch_size=project["trainConfig"]["batch_size"],
                start_epoch=0, 
                epoch=project["trainConfig"]["epochs"],
                train_split=project["trainConfig"]["train_split"], 
                model_type=modelType, 
                model_weight=None,
                validate_matrix='val_acc',
                save_method=project["trainConfig"]["saveMethod"],
                step_lr=(150, 200),
                labels=model_label,
                weight_decay=5e-4,
                warm_up_epoch=6
            )
            if res:
                STAGE = 3
        # 4 ========== trained ========= #
    except Exception as e:
        print("Error : ", str(e))
        q.announce({"time":time.time(), "event": "error", "msg" : str(e)})        
    finally:
        print("Thread ended")



@app.route("/terminate_training", methods=["POST"])
def terminate_training():
    global train_task, reporter
    print("terminate current training process")
    if train_task and train_task.is_alive():
        train_task.join()
    return jsonify({"result" : "OK"})


@app.route("/inference_image", methods=["POST"])
def handle_inference_model():
    global STAGE, current_model
    if 'image' not in request.files:
        return "No image"
    if STAGE < 3:
        return "Training not success yet :" + str(STAGE)
    
    tmp_img = request.files['image']
    project_id = request.form['project_id']
    model_type = request.form['type']

    if not tmp_img:
        return "Image null or something"
    
    target_file_path = os.path.join(PROJECT_PATH, project_id, TEMP_FOLDER)
    helper.create_not_exist(target_file_path) 
    target_file = os.path.join(target_file_path, tmp_img.filename)
    tmp_img.save(target_file)    

    if model_type == "classification":
        orig_image, img = helper.prepare_image(target_file, current_model, current_model.input_size)
        elapsed_ms, prob, prediction = current_model.predict(img)
        return jsonify({"result" : "OK","prediction":prediction, "prob":np.float64(prob)})
    elif model_type == "detection":
        threshold = float(request.form['threshold'])
        orig_image, input_image = helper.prepare_image(target_file, current_model, current_model._input_size)
        height, width = orig_image.shape[:2]
        prediction_time, boxes, probs = current_model.predict(input_image, height, width, threshold)
        labels = current_model._labels
        bboxes = []
        for box, classes in zip(boxes, probs):
            x1, y1, x2, y2 = box
            bboxes.append({
                "x1" : np.float64(x1), 
                "y1" : np.float64(y1), 
                "x2" : np.float64(x2), 
                "y2" : np.float64(y2), 
                "prob" : np.float64(classes.max()), 
                "label" : labels[np.argmax(classes)]
            })
        return jsonify({"result" : "OK", "boxes": bboxes})
    else:
        return jsonify({"result" : "FAIL","reason":"model type not specify"})


if __name__ == '__main__':
    print("BACKEND : " + BACKEND)
    print("DEVICE : " + DEVICE)
    len_arg = len(sys.argv)
    if len_arg > 2:
        if sys.argv[1] == "tuna" and sys.argv[2]:
            print("=== start tuna ===")
            tuna.run_tuna(5000,sys.argv[2])

    app.run(host="0.0.0.0",debug=True)
