const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();

app.use(cors());

app.use(express.json());

const MODEL_PATH = 'model/model.json';
let model;

async function loadModel() {
  model = await tf.loadLayersModel('file://' + MODEL_PATH);
  console.log('Model loaded successfully');
}

loadModel();

app.use('/images', express.static(__dirname + '/public/images'));

app.post('/classify', async (req, res) => {
  try {
    const imgPath = path.join(__dirname, 'public', 'images', req.body.imageName);
    const imageBuffer = fs.readFileSync(imgPath);
    const tfimage = tf.node.decodeImage(imageBuffer);
    const tfresized = tf.image.resizeBilinear(tfimage, [224, 224]);
    const tfexpanded = tfresized.expandDims(0);
    const tfnormalized = tfexpanded.div(255.0);
    const predictions = model.predict(tfnormalized);
    const top3 = Array.from(predictions.dataSync())
      .map((score, index) => ({ score, index }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
      .map(({ score, index }) => ({
        class: getClass(index),
        score: Math.round(score * 100)
      }));

    res.send(top3);
  } catch (err) {
    console.log(err);
    res.sendStatus(500);
  }
});

function getClass(index) {
  const classes = require('./model/skin_classes.json');
  return classes[index];
}

app.use(express.static(__dirname + '/public'));

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server started at port ${PORT}`);
});
