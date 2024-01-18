import React, { useState } from "react";
import "./styles/index.css";
import SpotifyApp from "./SpotifyApp";
import HandGestures from "./HandGestures";

/**
 * Main application component that renders the SpotifyApp and HandGestures components.
 *
 * @component
 * @returns {JSX.Element} - The rendered App component.
 */
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
