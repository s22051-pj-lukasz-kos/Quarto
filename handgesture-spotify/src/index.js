import React from "react";
import ReactDOM from "react-dom";
import "./index.css";
import App from "./App";

// const root = ReactDOM.createRoot(document.getElementById("root"));
// root.render(
//   <React.StrictMode>
//     <App />
//   </React.StrictMode>
// );
// DON'T CHANGE THIS, and keep react on version 17. Based on:
// https://community.spotify.com/t5/Spotify-for-Developers/Spotify-Web-Playback-SDK-example-playback-buttons-don-t-work/m-p/5518053#M8262
ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById("root")
);
