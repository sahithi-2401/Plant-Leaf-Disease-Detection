import os
from flask import Flask, redirect, render_template, request, session
from PIL import Image
import torchvision.transforms.functional as TF
import CNN as RESNET50
import numpy as np
import torch
import pandas as pd
from deep_translator import GoogleTranslator

# Load data
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load model
model = RESNET50.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()


def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session


@app.route('/')
def home_page():
    return render_template('home.html')


@app.route('/contact')
def contact():
    return render_template('contact-us.html')


@app.route('/index')
def ai_engine_page():
    return render_template('index.html')


@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')


@app.route('/social')
def social():
    return render_template('social.html')


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        # Process new image upload
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)

        pred = prediction(file_path)

        # Store prediction ID in session for later use
        session['last_pred'] = int(pred)
        session['last_image_path'] = image_url = disease_info['image_url'][pred]

        # Get language preference
        target_lang = request.form.get('lang', 'en')

    else:  # GET method (language translation)
        # Get the language to translate to
        target_lang = request.args.get('lang', 'en')

        # If we don't have a prediction stored, redirect to home
        if 'last_pred' not in session:
            return redirect('/')

        # Get the stored prediction
        pred = session['last_pred']
        image_url = session.get(
            'last_image_path', disease_info['image_url'][pred])

    # Get all the information based on prediction
    title = disease_info['disease_name'][pred]
    description = disease_info['description'][pred]
    prevent = disease_info['Possible Steps'][pred]
    if 'image_url' not in locals():
        image_url = disease_info['image_url'][pred]
    supplement_name = supplement_info['supplement name'][pred]
    supplement_image_url = supplement_info['supplement image'][pred]
    supplement_buy_link = supplement_info['buy link'][pred]

    # Translate if needed
    if target_lang != 'en':
        try:
            translator = GoogleTranslator(source='en', target=target_lang)
            title = translator.translate(title)
            description = translator.translate(description)
            prevent = translator.translate(prevent)
            supplement_name = translator.translate(supplement_name)
        except Exception as e:
            print(f"Translation error: {e}")

    return render_template('submit.html',
                           title=title,
                           desc=description,
                           prevent=prevent,
                           image_url=image_url,
                           pred=pred,
                           sname=supplement_name,
                           simage=supplement_image_url,
                           buy_link=supplement_buy_link)


@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html',
                           supplement_image=list(
                               supplement_info['supplement image']),
                           supplement_name=list(
                               supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']),
                           buy=list(supplement_info['buy link']))


if __name__ == '__main__':
    app.run(debug=True)
