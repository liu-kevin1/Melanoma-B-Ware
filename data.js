// blank mustafo 
const tf = require('@tensorflow/tfjs');
const tf_converter = require('@tensorflow/tfjs-converter');

const MODEL_URL = 'model-to-js/model.json';

const model = tf_converter.loadGraphModel(MODEL_URL);
// const cat = document.getElementById('cat');
// model.execute(tf.browser.fromPixels(cat));
// const example = 
var submit = document.getElementById("submit");
submit.onclick = function() {
    var message = document.getElementById("message").value;
    
    var output = model.predict(message);

    document.getElementById('output').innerText = output;    
}

