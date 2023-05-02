const express = require("express");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const cors = require("cors");
const axios = require("axios");
const https = require("https");

const app = express();
app.use(express.json());
app.use(cors());

const MODEL_PATH = "model/model.json";
let model;

async function loadModel() {
  try {
    model = await tf.loadLayersModel("file://" + MODEL_PATH);
    console.log("Model loaded successfully");
  } catch (error) {
    console.error("Error loading model:", error);
  }
}

loadModel();

const privateKey = fs.readFileSync(
  "/etc/letsencrypt/live/clasifier.iderm4u.com/privkey.pem",
  "utf8"
);
const certificate = fs.readFileSync(
  "/etc/letsencrypt/live/clasifier.iderm4u.com/fullchain.pem",
  "utf8"
);
const credentials = { key: privateKey, cert: certificate };

app.use("/images", express.static(__dirname + "/public/images"));

app.post("/classify", async (req, res) => {
  try {
    console.log("Image URL:", req.body.imageUrl);
    const response = await axios.get(req.body.imageUrl, {
      responseType: "arraybuffer",
    });
    console.log("Content-Type:", response.headers["content-type"]);
    const imageBuffer = Buffer.from(response.data, "binary");
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
        score: Math.round(score * 100),
      }));

    res.send(top3);
  } catch (error) {
    console.error("Error during classification:", error.message);
    res.sendStatus(500);
  }
});

function getClass(index) {
  const classes = require("model/skin_classes.json");
  return classes[index];
}

app.use(express.static(__dirname + "/public"));

const httpsServer = https.createServer(credentials, app);

httpsServer.listen(443, () => {
  console.log("HTTPS Server running on port 443");
});
