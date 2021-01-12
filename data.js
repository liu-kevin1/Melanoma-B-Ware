// blank mustafo 
const tf = require('@tensorflow/tfjs');
const tf_converter = require('@tensorflow/tfjs-converter');

const MODEL_URL = 'model-to-js/model.json';

const model = tf_converter.loadGraphModel(MODEL_URL);
const cat = document.getElementById('cat');
model.execute(tf.browser.fromPixels(cat));