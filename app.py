from flask import Flask,render_template,request
import tensorflow as tf
import numpy as np
from PIL import Image,ImageOps
from tensorflow.keras.models import load_model


app=Flask(__name__)
model=load_model('fashion_model.hdf5')

def classify(image):
    img=ImageOps.grayscale(image)
    img=ImageOps.invert(img)
    img=img.resize((28,28))
    img=np.array(img).astype('float32')/255.0
    img=img.reshape(1,28,28,1)
    return img
@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        file=request.files['file']
        if file:
            class_name=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
            image=Image.open(file)
            img=classify(image)
            pred=model.predict(img)
            pred=np.argmax(pred,axis=1)
            pred_cls=class_name[pred[0]]
            return render_template('result.html',pred_cls=pred_cls)
    return render_template('index.html')
if __name__=='__main__':
    app.run(debug=True,use_reloader=False)