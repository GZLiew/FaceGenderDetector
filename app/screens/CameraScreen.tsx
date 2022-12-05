import React, { useEffect, useState } from "react"
import { Camera, CameraType, FaceDetectionResult } from "expo-camera"
import * as FaceDetector from "expo-face-detector"
import * as FileSystem from "expo-file-system"
import { StyleSheet, View, Text } from "react-native"
import * as tf from "@tensorflow/tfjs"
import { bundleResourceIO, decodeJpeg } from "@tensorflow/tfjs-react-native"
const modelJSON = require("../tfmodels/face/model.json")
const modelWeight1 = require("../tfmodels/face/group1-shard1of10.bin")
const modelWeight2 = require("../tfmodels/face/group1-shard2of10.bin")
const modelWeight3 = require("../tfmodels/face/group1-shard3of10.bin")
const modelWeight4 = require("../tfmodels/face/group1-shard4of10.bin")
const modelWeight5 = require("../tfmodels/face/group1-shard5of10.bin")
const modelWeight6 = require("../tfmodels/face/group1-shard6of10.bin")
const modelWeight7 = require("../tfmodels/face/group1-shard7of10.bin")
const modelWeight8 = require("../tfmodels/face/group1-shard8of10.bin")
const modelWeight9 = require("../tfmodels/face/group1-shard9of10.bin")
const modelWeight10 = require("../tfmodels/face/group1-shard10of10.bin")

const modelWeights = [
  modelWeight1,
  modelWeight2,
  modelWeight3,
  modelWeight4,
  modelWeight5,
  modelWeight6,
  modelWeight7,
  modelWeight8,
  modelWeight9,
  modelWeight10,
]

const loadModel = async () => {
  //.ts: const loadModel = async ():Promise<void|tf.LayersModel>=>{
  try {
    await tf.ready()
    const model = await tf.loadLayersModel(bundleResourceIO(modelJSON, modelWeights))
    console.log("model loaded")
    return model
  } catch (e) {
    console.log("[LOADING ERROR] info:", e)
    return null
  }
}

const transformImageToTensor = async (uri: string) => {
  //.ts: const transformImageToTensor = async (uri:string):Promise<tf.Tensor>=>{
  //read the image as base64
  const img64 = await FileSystem.readAsStringAsync(uri, {
    encoding: FileSystem.EncodingType.Base64,
  })
  const imgBuffer = tf.util.encodeString(img64, "base64").buffer
  const raw = new Uint8Array(imgBuffer)
  let imgTensor = decodeJpeg(raw)
  // const scalar = tf.scalar(255)
  //resize the image
  imgTensor = tf.image.resizeNearestNeighbor(imgTensor, [80, 80])
  //normalize; if a normalization layer is in the model, this step can be skipped
  // const tensorScaled = imgTensor.div(scalar)
  //final shape of the tensor
  const img = tf.image.resizeBilinear(imgTensor, [80, 80], false)
  return img
}

const makePredictions = async (batch: any, model: any, imagesTensor: any) => {
  //.ts: const makePredictions = async (batch:number, model:tf.LayersModel,imagesTensor:tf.Tensor<tf.Rank>):Promise<tf.Tensor<tf.Rank>[]>=>{
  //cast output prediction to tensor
  const predictionsdata = model.predict(imagesTensor)
  //.ts: const predictionsdata:tf.Tensor = model.predict(imagesTensor) as tf.Tensor
  let pred = predictionsdata.split(batch) //split by batch size
  //return predictions
  return pred
}

export const getPredictions = async (image: any, model: tf.LayersModel) => {
  const tensor_image = await transformImageToTensor(image)
  const predictions = await makePredictions(1, model, tensor_image)
  return predictions
}

const styles = StyleSheet.create({
  root: {
    width: "100%",
    height: "100%",
  },
})

export const CameraScreen = () => {
  const [cameraLoaded, setCameraLoaded] = useState(false)
  const cameraRef = React.useRef<Camera>()
  const tfModelRef = React.useRef<tf.LayersModel>(null)
  const detectedAgeAndGender = React.useRef(false)

  const [status] = Camera.useCameraPermissions()
  Camera.useMicrophonePermissions()

  useEffect(() => {
    setCameraLoaded(false)
    tf.ready()
      .then(() => {
        return loadModel()
      })
      .then((model) => {
        tfModelRef.current = model
        setCameraLoaded(true)
      })
  }, [])

  const handleFacesDetected: any = async ({ faces }: FaceDetectionResult) => {
    console.log(faces)

    if (detectedAgeAndGender.current || !faces || faces.length === 0) {
      console.log("exit early")
      return
    }

    let imageUri: string = null
    try {
      const image = await cameraRef.current.takePictureAsync()
      imageUri = image.uri
      if (!imageUri) {
        return
      }
      detectedAgeAndGender.current = true
      const predictions = await getPredictions(imageUri, tfModelRef.current)
      console.log(JSON.stringify(predictions, null, 2))
      // getPredictions(imageUri)
    } catch (error) {
      console.log(error)
    } finally {
      // await FileSystem.deleteAsync(imageUri)
    }
  }

  return (
    <View style={[styles.root]}>
      {!cameraLoaded && <Text>Loading camera...</Text>}
      {cameraLoaded && status?.granted && (
        <Camera
          ref={cameraRef}
          onFacesDetected={handleFacesDetected}
          type={CameraType.front}
          faceDetectorSettings={{
            mode: FaceDetector.FaceDetectorMode.fast,
            detectLandmarks: FaceDetector.FaceDetectorLandmarks.none,
            runClassifications: FaceDetector.FaceDetectorClassifications.none,
            minDetectionInterval: 5000,
            tracking: true,
          }}
          style={StyleSheet.absoluteFill}
        />
      )}
    </View>
  )
}
