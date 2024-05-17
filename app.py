from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

dic = {0 : 'Papper',
       1 : 'Rock',
       2 : 'Scissors',
       }

model = load_model('model_kertas.h5')

model.make_predict_function()

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(100,150))
    i = image.img_to_array(i)
    i = i.reshape((100,150,3))
    i = np.expand_dims(i, axis=0)
    predictions = model.predict(i)
    predicted_class = predictions.argmax()
    
    if predictions.ndim > 1:
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class]
        return dic[predicted_class], confidence
    else:
        return "Tidak ada prediksi yang tersedia", 0

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Please subscribe Artificial Intelligence Hub..!!!"

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename    
        img.save(img_path)
        p = predict_label(img_path)

    return render_template("index.html", prediction=p, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
