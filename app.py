import os
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask,render_template,request
from tensorflow.keras.preprocessing.image import load_img , img_to_array

app = Flask(__name__)
model = load_model('model.hdf5')

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

classes = ['COVID-19','HEALTHY','LUNG CANCER','PNEUMONIA']

def predict(Image):
    img = load_img(Image,target_size=(224,224))
    img = img_to_array(img)
    img = img/255.0
    img = np.expand_dims(img, axis = 0) 
    prediction = model.predict(img)
    dict_result = {}
    for i in range(4):
        dict_result[prediction[0][i]] = classes[i]

    result = prediction[0]
    result.sort()
    result = result[::-1]
    probability = result[:4]
    
    prob_result = []
    class_result = []
    for i in range(4):
        prob_result.append((probability[i]*100).round(2))
        class_result.append(dict_result[probability[i]])

    return class_result,prob_result

@app.route('/')
def home():
        return render_template("index.html")

@app.route('/results' , methods = ['GET' , 'POST'])
def results():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':     
        if (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img,file.filename))
                img_path = os.path.join(target_img,file.filename)
                img = file.filename

                class_result , prob_result = predict(img_path)

                predictions = {
                    "class1":class_result[0],
                    "class2":class_result[1],
                    "class3":class_result[2],
                    "class4":class_result[3],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                    "prob4": prob_result[3],
                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('results.html',img=img,predictions=predictions)
            else:
                return render_template('index.html',error=error)

    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)