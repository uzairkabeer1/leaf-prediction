from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import os
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms

app = Flask(__name__, static_folder='static')


def load_pytorch_model(model_path):
    return torch.load(model_path)

def predict_user_image(image_path, model):
    preprocess = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    preprocessed_image = preprocess(img)
    batch_img_tensor = torch.unsqueeze(preprocessed_image, 0)

    model.eval()
    out = model(batch_img_tensor)
    out = str(out.argmax())

    p = {
        'tensor(0)': 'P0_diseased',
        'tensor(1)': 'P0_healthy',
        'tensor(2)': 'P10_diseased',
        'tensor(3)': 'P10_healthy',
        'tensor(4)': 'P11_diseased',
        'tensor(5)': 'P11_healthy',
        'tensor(6)': 'P3_diseased',
        'tensor(7)': 'P3_healthy',
        'tensor(8)': 'P5_diseased',
        'tensor(9)': 'P5_healthy',
        'tensor(10)': 'P6_diseased',
        'tensor(11)': 'P6_healthy',
        'tensor(12)': 'P7_diseased',
        'tensor(13)': 'P7_healthy',
        'tensor(14)': 'P9_diseased',
        'tensor(15)': 'P9_healthy'
    }

    out = p[out]
    plant, condition = out.split('_')

    plant_folders = {
        'Mango, XYZ': 'P0',
        'Guava':'P3',
        'Jamun':'P5',
        'Jatropha':'P6',
        'Pongamia Pinnata':'P7',
        'Pomegranate':'P9',
        'Lemon':'P10',
        'Chinar':'P11'
    }

    plant_name = {v: k for k, v in plant_folders.items()}
    return plant_name[plant], condition

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        print(filename)
        folder_path = 'uploaded_images'
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        f.save(file_path)

        model_path = "leaf_Model.pth"
        loaded_model = load_pytorch_model(model_path)

        plant, condition = predict_user_image(file_path, loaded_model)
        return render_template('result.html', filename=filename, plant=plant, condition=condition)

    return render_template('index.html')

@app.route('/uploaded_images/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploaded_images', filename)

if __name__ == '__main__':
    app.run(debug=True)
