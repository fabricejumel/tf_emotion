import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from PIL import Image
from repvgg import create_RepVGG_A0 as create

# Load model
model = create(deploy=True)

# 8 Emotions
emotions = ("anger","contempt","disgust","fear","happy","neutral","sad","surprise")

def init(device):
    # Initialise model
    global dev
    dev = device
    model.to(device)
    model.load_state_dict(torch.load("weights/repvgg.pth"))

    # Save to eval
    cudnn.benchmark = True
    model.eval()

def detect_emotion(images, conf=True, min_size=10):
    with torch.no_grad():
        # Normalise and transform images
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        processed_images = []
        for image in images:
            try:
                # Convert the image and preprocess
                img = Image.fromarray(image)
                
                # Check the size of the image
                if img.size[0] < min_size or img.size[1] < min_size:
                    print(f"Skipping image with dimensions {img.size}, smaller than {min_size}x{min_size}.")
                    continue  # Skip the image if it's too small
                
                # Apply the preprocessing transformations
                processed_images.append(preprocess(img))
            except Exception as e:
                print(f"Error processing image: {e}")
        
        # Check if any images were processed
        if len(processed_images) == 0:
            print("No valid images to process.")
            return []  # Return an empty list if no images were processed

        # Stack the processed images for model input
        x = torch.stack(processed_images)
        
        # Feed through the model
        y = model(x.to(dev))
        
        result = []
        for i in range(y.size(0)):
            # Get the predicted emotion and its confidence
            emotion_idx = torch.argmax(y[i]).item()
            confidence = torch.softmax(y[i], dim=0)[emotion_idx].item() * 100
            
            # Append result with or without confidence
            emotion_label = emotions[emotion_idx]
            if conf:
                result.append([f"{emotion_label} ({confidence:.1f}%)", emotion_idx])
            else:
                result.append([emotion_label, emotion_idx])
                
    return result
