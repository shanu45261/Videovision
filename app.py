# app.py
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model definition
class VideoVisionAI(nn.Module):
    def __init__(self, num_classes=10, hidden_size=512):
        super(VideoVisionAI, self).__init__()
        
        # Load pretrained ResNet
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # CNN layers
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # RNN for temporal features
        self.rnn = nn.LSTM(
            input_size=1024,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self.feature_extractor(x)
        x = self.cnn_layers(x)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.rnn(x)
        x = self.classifier(x[:, -1, :])
        return x

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VideoVisionAI().to(device)
model.eval()

# Preprocessing transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def analyze_video(video_path):
    """Analyze video and return predictions."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < 16:  # Process first 16 frames
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame).unsqueeze(0)
        frames.append(frame)
    
    cap.release()
    
    if len(frames) < 16:
        return {"error": "Video too short"}
    
    # Stack frames and process
    frames = torch.cat(frames).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(frames.to(device))
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
    # Convert predictions to list
    predictions = probabilities[0].cpu().numpy().tolist()
    
    # Example categories (replace with your actual categories)
    categories = ["Action", "Comedy", "Drama", "Horror", "Romance", 
                 "Documentary", "Animation", "Thriller", "Sci-Fi", "Adventure"]
    
    results = {
        categories[i]: float(predictions[i])
        for i in range(len(categories))
    }
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            results = analyze_video(filepath)
            os.remove(filepath)  # Clean up uploaded file
            return jsonify(results)
        except Exception as e:
            os.remove(filepath)  # Clean up on error
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)