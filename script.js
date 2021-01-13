// blank mustafo 
// var requirejs = require(['requirejs']);

// require(['@tensorflow/tfjs'], function (foo) {
//     //foo is now loaded.
// });

// const tf = requirejs('@tensorflow/tfjs');
// const tf_converter = requirejs('@tensorflow/tfjs-converter');
//import * as tf from '@tensorflow/tfjs';

const MODEL_URL = '/model-to-js/model.json';

async function test() {
    const model = tf.loadLayersModel(MODEL_URL);
    return model;
}

// const cat = document.getElementById('cat');
// model.execute(tf.browser.fromPixels(cat));
// const example = 

// var submit = document.getElementById("submit");
// submit.onclick = function() {
//     var message = document.getElementById("message").value;
    
//     var output = model.predict(message);

//     document.getElementById('output').innerText = output;    
// }


var mod = test();
console.log(mod);