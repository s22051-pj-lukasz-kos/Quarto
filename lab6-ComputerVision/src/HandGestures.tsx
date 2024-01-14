import React, { useEffect, useRef, useState } from "react";
import "./styles/main.scss";
import * as mdc from "material-components-web";
import {
  GestureRecognizer,
  FilesetResolver,
  DrawingUtils,
} from "@mediapipe/tasks-vision";

function HandGestures({ setCategoryName }) {
  const [webcamRunning, setWebcamRunning] = useState<boolean>(false);
  // for showing section when loaded
  const [isGestureRecognizerLoaded, setIsGestureRecognizerLoaded] =
    useState<boolean>(false);
  const [gestureRecognizer, setGestureRecognizer] =
    useState<GestureRecognizer | null>(null);
  const videoRef = useRef(null);
  const canvasVideoRef = useRef(null);
  const gestureOutputRef = useRef(null);
  const webcamButtonRef = useRef(null);
  const [isVideoLoaded, setIsVideoLoaded] = useState<boolean>(false);
  const [isWebcamAllowed, setIsWebcamAllowed] = useState<boolean>(false);
  const [canvasCtx, setCanvasCtx] = useState(null);
  const [showVideoResults, setShowVideoResults] = useState<boolean>(false);
  const [gestureOutputResult, setGestureOutputResult] = useState<String>("");

  const videoHeight = "360px";
  const videoWidth = "480px";

  // sets canvasCtx to video canvas
  useEffect(() => {
    const videoElement = videoRef.current as HTMLVideoElement;
    const canvasElement = canvasVideoRef.current as HTMLCanvasElement;
    if (videoElement && isVideoLoaded && canvasElement) {
      setCanvasCtx(canvasElement.getContext("2d"));
    }
  }, [isVideoLoaded]);

  useEffect(() => {
    mdc.autoInit();
    // Before we can use HandLandmarker class we must wait for it to finish
    // loading. Machine Learning models can be large and take a moment to
    // get everything needed to run.
    const createGestureRecognizer = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        let response = await GestureRecognizer.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
        });
        setGestureRecognizer(response);
        setIsGestureRecognizerLoaded(true);
      } catch (error) {
        console.error("Error when creating GestureRecognizer:", error);
      }
    };
    createGestureRecognizer();

    const videoElement = videoRef.current;
    const canvasElement = canvasVideoRef.current;
    if (videoElement && canvasElement) {
      setIsVideoLoaded(true);
    }
    // Check if webcam access is supported.
    if (!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
      setIsWebcamAllowed(true);
    } else {
      console.warn("getUserMedia() is not supported by your browser");
    }
  }, []);

  const predictWebcam = async (event) => {
    if (!webcamRunning) return;

    let lastVideoTime = -1;
    let results = undefined;
    const videoElement = videoRef.current;
    const canvasElement = canvasVideoRef.current;
    let nowInMs = Date.now();
    if (videoElement.currentTime !== lastVideoTime) {
      lastVideoTime = videoElement.currentTime;
      results = gestureRecognizer.recognizeForVideo(videoElement, nowInMs);
    }
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    const drawingUtils = new DrawingUtils(canvasCtx);

    if (results.landmarks) {
      for (const landmarks of results.landmarks) {
        drawingUtils.drawConnectors(
          landmarks,
          GestureRecognizer.HAND_CONNECTIONS,
          {
            color: "#00FF00",
            lineWidth: 5,
          }
        );
        drawingUtils.drawLandmarks(landmarks, {
          color: "#FF0000",
          lineWidth: 2,
        });
      }
    }
    canvasCtx.restore();
    if (results.gestures.length > 0) {
      setShowVideoResults(true);
      const categoryName = results.gestures[0][0].categoryName;
      const categoryScore = (results.gestures[0][0].score * 100).toFixed(2);
      const handedness = results.handednesses[0][0].displayName;
      let gestureOutputInnerText = `GestureRecognizer: ${categoryName}\n Confidence: ${categoryScore} %\n Handedness: ${handedness}`;
      setCategoryName(categoryName);
      setGestureOutputResult(gestureOutputInnerText);
    }
    // Call this function again to keep predicting when the browser is ready.
    if (webcamRunning === true) {
      window.requestAnimationFrame(predictWebcam);
    }
  };

  const handleWebcamButton = () => {
    if (!isWebcamAllowed) {
      alert(
        "Webcam is not allowed. Set permission using icon on the right side of address bar."
      );
    }
    if (!gestureRecognizer) {
      alert("Please wait for gestureRecognizer to load");
      return;
    }
    if (webcamRunning) {
      // Stop the webcam stream
      const videoElement = videoRef.current;
      if (videoElement && videoElement.srcObject) {
        const stream = videoElement.srcObject;
        const tracks = stream.getTracks();

        tracks.forEach((track) => track.stop());
        videoElement.srcObject = null;
      }
      setWebcamRunning(false);
    } else {
      // Start the webcam stream
      const constraints = { video: true };
      navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        const videoElement = videoRef.current;
        if (videoElement) {
          videoElement.srcObject = stream;
        }
      });
      setWebcamRunning(true);
    }
  };

  const gestureOutputStyle = {
    display: "block",
    width: videoWidth,
  };

  return (
    <>
      <section
        id="demos"
        className={isGestureRecognizerLoaded ? "" : "invisible"}
      >
        <p>
          Click <b>enable predictions</b> below and grant access to the webcam
          if prompted.
        </p>
        <p>Manipulate Spotify by using 4 hand gestures</p>
        <ul>
          <li>Raised index finger - play music</li>
          <li>Closed fist - stop music</li>
          <li>Thumb up - next track</li>
          <li>Thumb down - previous track</li>
        </ul>
        <div id="liveView" className="videoView">
          <button
            ref={webcamButtonRef}
            id="webcamButton"
            className="mdc-button mdc-button--raised"
            onClick={handleWebcamButton}
          >
            <span className="mdc-button__ripple" />
            <span className="mdc-button__label">
              {webcamRunning ? "DISABLE PREDICTIONS" : "ENABLE PREDICTIONS"}
            </span>
          </button>
          <div style={{ position: "relative" }}>
            <video
              ref={videoRef}
              id="webcam"
              autoPlay
              playsInline
              onLoadedData={predictWebcam}
              style={{ height: videoHeight, width: videoWidth }}
            />
            <canvas
              ref={canvasVideoRef}
              className="output_canvas"
              id="output_canvas"
              width={1280}
              height={720}
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                height: videoHeight,
                width: videoWidth,
              }}
            />
            <p
              ref={gestureOutputRef}
              id="gesture_output"
              className="output"
              style={
                showVideoResults ? gestureOutputStyle : { display: "none" }
              }
            >
              {gestureOutputResult}
            </p>
          </div>
        </div>
      </section>
    </>
  );
}

export default HandGestures;
