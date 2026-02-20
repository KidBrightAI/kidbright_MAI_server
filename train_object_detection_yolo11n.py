import os
import shutil
import time
import cv2
import yaml
import torch
import xml.etree.ElementTree as ET

import sys
try:
    from ultralytics import YOLO
except ImportError:
    # Ensure ultralytics is installed. The prompt says sys.path.append("ultralytics")
    sys.path.append("ultralytics")
    try:
        from ultralytics import YOLO
    except ImportError:
        pass # Will raise error at runtime if ultralytics is genuinely missing

def convert_voc_to_yolo(project_dir, labels, train_split=80):
    dataset_dir = os.path.join(project_dir, "dataset")
    yolo_dir = os.path.join(project_dir, "yolo_dataset")
    
    # recreate yolo dataset folder Let's check existing ones.
    if os.path.exists(yolo_dir):
        shutil.rmtree(yolo_dir)
        
    images_train_dir = os.path.join(yolo_dir, "images", "train")
    images_val_dir = os.path.join(yolo_dir, "images", "val")
    labels_train_dir = os.path.join(yolo_dir, "labels", "train")
    labels_val_dir = os.path.join(yolo_dir, "labels", "val")
    
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    # get images id
    img_ids = []
    # Using Kidbright IDE default: images are likely defined in ImageSets/Main/train.txt
    train_txt_path = os.path.join(dataset_dir, "ImageSets", "Main", "train.txt")
    if os.path.exists(train_txt_path):
        with open(train_txt_path, "r") as f:
            for line in f:
                img_id = line.strip()
                if len(img_id) > 0:
                    img_ids.append(img_id)
    else:
        # Fallback: read directly from JPEGImages
        jpeg_dir = os.path.join(dataset_dir, "JPEGImages")
        if os.path.exists(jpeg_dir):
            for filename in os.listdir(jpeg_dir):
                if filename.endswith(".jpg"):
                    img_ids.append(filename[:-4])
                    
    # Split train/val
    import random
    random.shuffle(img_ids)
    split_idx = int(len(img_ids) * (train_split / 100.0))
    train_ids = img_ids[:split_idx]
    val_ids = img_ids[split_idx:]
    
    # Ensure there is at least something in validation if possible, else copy train to val
    if len(val_ids) == 0 and len(train_ids) > 0:
        val_ids = train_ids.copy()
        
    def process_split(split_ids, target_images_dir, target_labels_dir):
        for img_id in split_ids:
            img_path = os.path.join(dataset_dir, "JPEGImages", f"{img_id}.jpg")
            xml_path = os.path.join(dataset_dir, "Annotations", f"{img_id}.xml")
            
            if not os.path.exists(img_path) or not os.path.exists(xml_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w, _ = img.shape
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            yolo_bboxes = []
            for obj in root.iter('object'):
                name_elem = obj.find('name')
                if name_elem is None:
                    continue
                name = name_elem.text.lower().strip()
                if name not in labels:
                    continue
                cls_id = labels.index(name)
                
                xmlbox = obj.find('bndbox')
                if xmlbox is None:
                    continue
                    
                xmin = float(xmlbox.find('xmin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymin = float(xmlbox.find('ymin').text)
                ymax = float(xmlbox.find('ymax').text)
                
                # yolo normalized
                x_center = ((xmin + xmax) / 2.0) / w
                y_center = ((ymin + ymax) / 2.0) / h
                width = (xmax - xmin) / w
                height = (ymax - ymin) / h
                
                yolo_bboxes.append(f"{cls_id} {x_center} {y_center} {width} {height}")
                
            # Copy image
            dest_img_path = os.path.join(target_images_dir, f"{img_id}.jpg")
            shutil.copy(img_path, dest_img_path)
            
            # Write txt
            dest_txt_path = os.path.join(target_labels_dir, f"{img_id}.txt")
            with open(dest_txt_path, "w") as f:
                f.write("\n".join(yolo_bboxes))
                
    process_split(train_ids, images_train_dir, labels_train_dir)
    process_split(val_ids, images_val_dir, labels_val_dir)
    
    # Generate data.yaml
    names_dict = {i: name for i, name in enumerate(labels)}
    data_yaml = {
        "path": os.path.abspath(yolo_dir),
        "train": "images/train",
        "val": "images/val",
        "names": names_dict
    }
    
    yaml_path = os.path.join(yolo_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
        
    return yaml_path

def train_object_detection_yolo11n(project, path_to_save, project_dir, q,
        high_resolution=True, 
        multi_scale=True, 
        cuda=True, 
        learning_rate=1e-4, 
        batch_size=32, 
        start_epoch=0, 
        epoch=100,
        train_split=80, 
        model_type='slim_yolo_v2', 
        model_weight=None,
        validate_matrix='val_acc',
        save_method='best',
        step_lr=(150, 200),
        labels=None,
        momentum=0.9,
        weight_decay=5e-4,
        warm_up_epoch=6
    ):
    
    os.makedirs(path_to_save, exist_ok=True)
    
    print("----------------------------------------------------------")
    print('Start dataset converting to YOLO format...')
    q.announce({"time":time.time(), "event": "dataset_loading", "msg" : "Loading and converting to YOLO dataset format..."})
    
    labels_lower = [l.lower() for l in labels]
    
    yaml_path = convert_voc_to_yolo(project_dir, labels_lower, train_split)
    
    print('Dataset formatted completely at', yaml_path)
    q.announce({"time":time.time(), "event": "dataset_loading", "msg" : "Dataset converted successfully."})
    print("----------------------------------------------------------")

    # Load a yolo11n model
    # User specified "yolo11n.pt" or model_type could carry the actual file needed, but let's default to yolo11n.pt
    model_file = "yolo11n.pt" if model_weight is None else model_weight
    if not os.path.exists(model_file):
        # Allow ultralytics to download it if it's "yolo11n.pt"
        # We can just pass the string to YOLO natively
        pass

    try:
        model = YOLO("yolo11n.pt")
    except Exception as e:
        q.announce({"time":time.time(), "event": "error", "msg" : f"Could not initialize YOLO model: {e}"})
        return False

    q.announce({"time":time.time(), "event": "train_start", "msg" : "Start training ..."})

    # Set up callbacks
    def on_train_epoch_start(trainer):
        q.announce({
            "time": time.time(), 
            "event": "epoch_start", 
            "msg": f"Start epoch {trainer.epoch + 1}/{trainer.epochs} ... training", 
            "epoch": trainer.epoch + 1, 
            "max_epoch": trainer.epochs
        })

    def on_train_batch_start(trainer):
        max_batch = len(trainer.train_loader) if hasattr(trainer, 'train_loader') else 0
        batch_i = getattr(trainer, 'batch_i', 0)
        q.announce({
            "time": time.time(), 
            "event": "batch_start", 
            "msg": f"Start batch {batch_i}/{max_batch} ... training",
            "batch": batch_i,
            "max_batch": max_batch
        })
        
    def on_train_batch_end(trainer):
        max_batch = len(trainer.train_loader) if hasattr(trainer, 'train_loader') else 0
        batch_i = getattr(trainer, 'batch_i', 0)
        
        train_loss = 0.0
        if hasattr(trainer, 'tloss'):
            try:
                # trainer.tloss is usually a tensor containing the loss items (box, cls, dfl), we can sum it to get total batch loss
                train_loss = float(trainer.tloss.sum())
            except:
                pass

        q.announce({
            "time": time.time(), 
            "event": "batch_end", 
            "msg": f"End batch {batch_i}/{max_batch} ... training",
            "batch": batch_i,
            "max_batch": max_batch,
            "matric": {
                "train_loss": train_loss
            }
        })

    def on_fit_epoch_end(trainer):
        metrics = getattr(trainer, 'metrics', {})
        # Keys in metrics can vary, try a few common patterns Ultralytics uses
        val_acc = metrics.get('metrics/mAP50(B)', metrics.get('val/mAP50(B)', 0.0))
        val_precision = metrics.get('metrics/precision(B)', metrics.get('val/precision(B)', 0.0))
        val_recall = metrics.get('metrics/recall(B)', metrics.get('val/recall(B)', 0.0))
        
        train_loss = 0.0
        if hasattr(trainer, 'tloss'):
            try:
                train_loss = float(trainer.tloss.sum())
            except:
                pass

        q.announce({
            "time": time.time(), 
            "event": "epoch_end", 
            "msg": f"End epoch {trainer.epoch + 1}/{trainer.epochs} ... training",
            "epoch": trainer.epoch + 1,
            "max_epoch": trainer.epochs,
            "matric": {
                "train_loss": train_loss,
                "val_acc": float(val_acc),                
                "val_precision": float(val_precision),
                "val_recall": float(val_recall),            
            }
        })

    def on_train_end(trainer):
        q.announce({"time": time.time(), "event": "train_end", "msg": "Training is done"})

    # clear existing if any and register
    model.clear_callback("on_train_epoch_start")
    model.clear_callback("on_train_batch_start")
    model.clear_callback("on_train_batch_end")
    model.clear_callback("on_fit_epoch_end")
    model.clear_callback("on_train_end")
    
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_start", on_train_batch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    device = 0 if cuda else "cpu"
    imgsz = 640 if high_resolution else 416
    
    try:
        # Run YOLO training
        results = model.train(
            data=yaml_path,
            epochs=epoch,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            lr0=learning_rate,
            project=path_to_save,
            name="yolo11n_run",
            exist_ok=True, # allow overwriting previous attempt
            verbose=True
        )
        
        best_pt_path = os.path.join(path_to_save, "yolo11n_run", "weights", "best.pt")
        target_pt_path = os.path.join(path_to_save, "best_map.pth")
        
        if os.path.exists(best_pt_path):
            shutil.copy(best_pt_path, target_pt_path)
            print("Successfully copied best.pt to best_map.pth")
            
    except Exception as e:
        q.announce({"time":time.time(), "event": "error", "msg" : f"Training failed: {e}"})
        return False
        
    print('Training is done')
    return True
