import React, { useState } from "react";
import "./styles/index.css";
import SpotifyApp from "./SpotifyApp";
import HandGestures from "./HandGestures";

function App() {
  const [categoryName, setCategoryName] = useState("");

  return (
    <React.StrictMode>
      <SpotifyApp categoryName={categoryName} />
      <HandGestures setCategoryName={setCategoryName} />
    </React.StrictMode>
  );
}

export default App;
