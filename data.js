// blank mustafo 
var requirejs = require(['requirejs']);
const tf = requirejs('@tensorflow/tfjs');
const tf_converter = requirejs('@tensorflow/tfjs-converter');
console.log(tf);
const MODEL_URL = 'model-to-js/model.json';

const model = tf_converter.loadGraphModel(MODEL_URL);
const cat = document.getElementById('cat');
model.execute(tf.browser.fromPixels(cat));