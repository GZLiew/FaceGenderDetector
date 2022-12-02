import React, { useEffect } from "react"
import { Camera, CameraType, FaceDetectionResult } from "expo-camera"
import * as FaceDetector from "expo-face-detector"
import * as FileSystem from "expo-file-system"
import { StyleSheet, View } from "react-native"
import * as tf from "@tensorflow/tfjs"
import { bundleResourceIO, decodeJpeg } from "@tensorflow/tfjs-react-native"
const modelJSON = require("../tfmodels/model.json")
const modelWeights = require("../tfmodels/group1-shard1of1.bin")

const loadModel = async () => {
  //.ts: const loadModel = async ():Promise<void|tf.LayersModel>=>{
  console.log(modelJSON)
  await tf.ready()
  const model = await tf
    .loadLayersModel(bundleResourceIO(modelJSON, modelWeights))
    .then((res) => console.log("model loaded", res))
    .catch((e) => {
      console.log("[LOADING ERROR] info:", e)
      console.trace()
    })
  return model
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
  const scalar = tf.scalar(255)
  //resize the image
  imgTensor = tf.image.resizeNearestNeighbor(imgTensor, [300, 300])
  //normalize; if a normalization layer is in the model, this step can be skipped
  const tensorScaled = imgTensor.div(scalar)
  //final shape of the rensor
  const img = tf.reshape(tensorScaled, [1, 300, 300, 3])
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

export const getPredictions = async (image: any) => {
  await tf.ready()
  const model = (await loadModel()) as tf.LayersModel
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
  const cameraRef = React.useRef<Camera>()
  const detectedAgeAndGender = React.useRef(false)

  const [status] = Camera.useCameraPermissions()
  Camera.useMicrophonePermissions()

  const handleFacesDetected: any = async ({ faces }: FaceDetectionResult) => {
    console.log(faces)

    if (detectedAgeAndGender.current || !faces || faces.length === 0) {
      console.log("exit early")
      return
    }

    let imageUri: string = null
    try {
      // const image = await cameraRef.current.takePictureAsync()
      // imageUri = image.uri
      // console.log(image.uri)
      detectedAgeAndGender.current = true
      await loadModel()
      // getPredictions(imageUri)
    } catch (error) {
      console.log(error)
    } finally {
      // await FileSystem.deleteAsync(imageUri)
    }
  }

  return (
    <View style={[styles.root]}>
      {status?.granted && (
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
