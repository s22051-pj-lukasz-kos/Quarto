import React, { useCallback, useEffect, useRef, useState } from "react";
import "./styles/main.scss";
import * as mdc from "material-components-web";
import {
  GestureRecognizer,
  FilesetResolver,
  DrawingUtils,
} from "@mediapipe/tasks-vision";

type RunningMode = "IMAGE" | "VIDEO";

function HandGestures() {
  const [webcamRunning, setWebcamRunning] = useState<boolean>(false);
  // for showing section when loaded
  const [isGestureRecognizerLoaded, setIsGestureRecognizerLoaded] =
    useState<boolean>(false);
  const [gestureRecognizer, setGestureRecognizer] =
    useState<GestureRecognizer | null>(null);
  const [runningMode, setRunningMode] = useState<RunningMode>("IMAGE");
  const [isInfoVisible, setIsInfoVisible] = useState<boolean>(false);
  const [infoInnerText, setInfoInnerText] = useState("");
  const [eventTarget, setEventTarget] = useState(null);
  const [isCanvasVisible, setIsCanvasVisible] = useState<boolean>(false);
  const canvasImageRef = useRef(null);
  const videoRef = useRef(null);
  const canvasVideoRef = useRef(null);
  const gestureOutputRef = useRef(null);
  const webcamButtonRef = useRef(null);
  const [isVideoLoaded, setIsVideoLoaded] = useState<boolean>(false);
  const [isWebcamAllowed, setIsWebcamAllowed] = useState<boolean>(false);
  const [canvasCtx, setCanvasCtx] = useState(null);
  const [imageResults, setImageResults] = useState(null);
  const [isStreamReady, setIsStreamReady] = useState<boolean>(false);
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
          runningMode: "IMAGE",
        });
        setGestureRecognizer(response);
        setIsGestureRecognizerLoaded(true);
      } catch (error) {
        console.error("Error when creating GestureRecognizer:", error);
      }
    };
    createGestureRecognizer();

    /********************************************************************
  // Demo 2: Continuously grab image from webcam stream and detect it.
  ********************************************************************/

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

  const handleImageClick = async (event) => {
    if (!gestureRecognizer) {
      alert("Please wait for gestureRecognizer to load");
      return;
    }

    if (runningMode === "VIDEO") {
      setRunningMode("IMAGE");
      await gestureRecognizer.setOptions({ runningMode: "IMAGE" });
    }

    const results = gestureRecognizer.recognize(event.target);

    // View results in the console to see their format
    console.log(results);
    if (results.gestures.length > 0) {
      setIsInfoVisible(true);
      let categoryName = results.gestures[0][0].categoryName;
      let categoryScore = (results.gestures[0][0].score * 100).toFixed(2);
      let handedness = results.handednesses[0][0].displayName;
      setInfoInnerText(
        `GestureRecognizer: ${categoryName}\n Confidence: ${categoryScore}%\n Handedness: ${handedness}`
      );
      setImageResults(results);
      setEventTarget(event.target);
      setIsCanvasVisible(true);
    }
  };

  useEffect(() => {
    if (isCanvasVisible) {
      const canvasElement = canvasImageRef.current as HTMLCanvasElement;
      if (canvasElement) {
        const canvasCtx = canvasElement.getContext("2d");
        const drawingUtils = new DrawingUtils(canvasCtx);
        for (const landmarks of imageResults.landmarks) {
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
            lineWidth: 1,
          });
        }
      }
    }
  }, [isCanvasVisible]);

  const predictWebcam = async (event) => {
    let lastVideoTime = -1;
    let results = undefined;
    const webcamElement = videoRef.current;
    const videoElement = videoRef.current;
    const canvasElement = canvasVideoRef.current;
    const gestureOutput = gestureOutputRef.current;
    // Now let's start detecting the stream.
    if (runningMode === "IMAGE") {
      setRunningMode("VIDEO");
      await gestureRecognizer.setOptions({ runningMode: "VIDEO" });
    }
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
    webcamRunning ? setWebcamRunning(false) : setWebcamRunning(true);

    // getUsermedia parameters.
    const constraints = {
      video: true,
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
      const videoElement = videoRef.current;
      videoElement.srcObject = stream;
      setIsStreamReady(true);
    });
  };

  const pStyle = {
    left: "0px",
    top: eventTarget !== null ? `${eventTarget.height}px` : "0px",
    width: eventTarget !== null ? `${eventTarget.width}px` : "0px",
  };

  const canvasStyle = {
    left: "0px",
    top: "0px",
    width: eventTarget !== null ? `${eventTarget.width}px` : "0px",
    height: eventTarget !== null ? `${eventTarget.height}px` : "0px",
  };

  const gestureOutputStyle = {
    display: "block",
    width: videoWidth,
  };

  const renderImageContainer = (src, alt) => {
    return (
      <div className="detectOnClick" onClick={handleImageClick}>
        <img
          src={src}
          alt={alt}
          crossOrigin="anonymous"
          title="Click to get recognize!"
        />
        <p
          className={isInfoVisible ? "info" : "classification removed"}
          style={infoInnerText !== "" ? pStyle : {}}
        >
          {infoInnerText}
        </p>
        <canvas
          ref={canvasImageRef}
          className={isCanvasVisible ? "canvas" : "removed"}
          width={eventTarget !== null ? `${eventTarget.naturalWidth}px` : "0px"}
          height={
            eventTarget !== null ? `${eventTarget.naturalHeight}px` : "0px"
          }
          style={eventTarget !== null ? canvasStyle : {}}
        ></canvas>
      </div>
    );
  };

  return (
    <>
      <section
        id="demos"
        className={isGestureRecognizerLoaded ? "" : "invisible"}
      >
        <h2>Demo: Recognize gestures</h2>
        <p>
          <em>Click on an image below</em> to identify the gestures in the
          image.
        </p>
        {renderImageContainer(
          "https://assets.codepen.io/9177687/thumbs-up-ga409ddbd6_1.png",
          "Right hand with thumb up"
        )}
        <h2>
          <br />
          Demo: Webcam continuous hand gesture detection
        </h2>
        <p>
          Use your hand to make gestures in front of the camera to get gesture
          classification. <br />
          Click <b>enable webcam</b> below and grant access to the webcam if
          prompted.
        </p>
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
