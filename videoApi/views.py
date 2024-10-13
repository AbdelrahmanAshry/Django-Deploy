# views.py
import torch
import os
import cv2
from django.http import JsonResponse
from django.shortcuts import render
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose
import torch.nn as nn
import numpy as np
# Preprocess video function (You need to define this function based on your preprocessing needs)
def preprocess_video_with_action_detection(video_path, num_frames=16, frame_skip=2, transform=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    prev_frame = None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale for optical flow
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is None:
            prev_frame = gray_frame
            continue
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Compute magnitude of flow
        magnitude = np.linalg.norm(flow, axis=2)
        
        # Threshold to detect motion (you can adjust this value)
        motion_detected = np.mean(magnitude) > 2.0  # This threshold can be tuned
        
        if motion_detected:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frame = Image.fromarray(frame)  # Convert to PIL image
            if transform:
                frame = transform(frame)
            frames.append(frame)
        
        # Update previous frame
        prev_frame = gray_frame

    cap.release()

    # If not enough frames, pad with the last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])

    # Stack frames into a single tensor and adjust the shape
    video_tensor = torch.stack(frames)  # Shape: (num_frames, C, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # Shape: (C, T, H, W)

    return video_tensor

def preprocess_video(video_path, num_frames=16, frame_skip=12, transform=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    resize_transform = Compose([
        Resize((224, 224)),  # Resize frames to 224x224
        ToTensor()           # Convert to tensor
    ])
    # Iterate over frames, skipping based on frame_skip interval
    for i in range(0, frame_count, frame_skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (OpenCV loads frames as BGR by default)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a PIL image and apply transforms (if any)
        frame = Image.fromarray(frame)
        if transform:
            frame = transform(frame)
        else:
            # Convert the frame to a tensor
            frame = resize_transform(frame)
        frames.append(frame)

        # Stop once we have the required number of frames
        if len(frames) >= num_frames:
            break

    cap.release()

    # If not enough frames, pad with the last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])

    # Stack frames into a single tensor and adjust the shape
    video_tensor = torch.stack(frames)  # Shape: (num_frames, C, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # Shape: (C, T, H, W)
    
    #video_tensor=video_tensor.ToTensor()
    return video_tensor

@csrf_exempt
def video_upload_view(request):
    if request.method == 'POST':
        # Retrieve the files from the request
        video_file = request.FILES.get('video_file')
        model_weights = request.FILES.get('model_weights')
        additional_input = request.POST.get('additional_input', '')

        # Ensure both files are provided
        if video_file and model_weights:
            # Save video and weights to temporary paths
            video_path = f"/tmp/{video_file.name}"
            weights_path = f"/tmp/{model_weights.name}"
            
            with open(video_path, 'wb+') as f:
                for chunk in video_file.chunks():
                    f.write(chunk)
                    
            with open(weights_path, 'wb+') as f:
                for chunk in model_weights.chunks():
                    f.write(chunk)

            # Preprocess the video
            video_tensor = preprocess_video(video_path)
            #video_tensor=preprocess_video_with_action_detection(video_path)
            # Load the VideoMAE model and weights
            feature_extractor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics", torch_dtype=torch.float16)
            model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
            # Modify classifier for binary classification (2 classes)
            model.classifier = torch.nn.Linear(model.config.hidden_size, 2)
            # Load the pre-trained model weights with filtering of mismatching keys
            state_dict = torch.load(weights_path)
            #filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}

            # Load filtered state dict
            #model.load_state_dict(filtered_state_dict, strict=False)            
            model.load_state_dict(state_dict, strict=False)            

            model.eval()
            video_tensor = video_tensor.unsqueeze(0)

            video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # [batch, channels, frames, height, width]

            # Run inference on the video
            with torch.no_grad():
                #inputs = feature_extractor(video_tensor, return_tensors="pt")
                outputs = model(video_tensor)
                logits_per_video = torch.mean(outputs.logits, dim=1)  # Aggregate over the 2 outputs

                # Check the shape of logits_per_video
                #print("Shape of logits_per_video:", logits_per_video.shape)

                # Apply softmax only if logits_per_video has two dimensions (batch_size, num_classes)
                if logits_per_video.dim() == 2:
                    predicted_probs = torch.softmax(logits_per_video, dim=1)  # Softmax across classes
                else:
                # If it's 1D, that means it has been aggregated incorrectly, handle that case
                    predicted_probs = torch.sigmoid(logits_per_video)  # Use sigmoid for binary classification
                print(f"Logits:{logits_per_video}")
                
                if logits_per_video > 6.22 :
                    predClass = 0 # Softmax across classes
                else:
                # If it's 1D, that means it has been aggregated incorrectly, handle that case
                    predClass=1  # Use sigmoid for binary classification
                print(f"Class:{predClass}")
    
                logits_per_video=logits_per_video-6

                # Set a threshold for determining class
                threshold = 0.56

                # Determine predicted class based on the highest probability
                #predicted_class = torch.argmax(predicted_probs), dim=1)  # Get index of the class with max probability

                # Calculate predicted class
                
                predicted_probs2 = torch.sigmoid(logits_per_video)
                predicted_class = (predicted_probs2 >= threshold).float()  # Use the dynamic threshold
                #logits = outputs.logits
                #predicted_class = torch.argmax(logits, dim=-1).item()
            predicted_class_int = int(predicted_class.item())  # Use .item() to get the value as a Python float, then convert to int

            # Map predicted class to label (shoplifter/safe)
            print(f"Class:{predicted_class_int}")
            class_map = {0: 'Safe', 1: 'Shoplifter'}
            result_label = class_map.get(predClass, 'Unknown')

            # Clean up temp files
            os.remove(video_path)
            os.remove(weights_path)

            # Return the result and render an animation based on the prediction
            return render(request, 'result.html', {'result_label': result_label})

        return JsonResponse({'error': 'Missing video or model files'}, status=400)

    return render(request, 'upload.html')
