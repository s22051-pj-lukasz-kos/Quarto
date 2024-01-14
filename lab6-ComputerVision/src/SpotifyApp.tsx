import React, { useState, useEffect } from "react";
import WebPlayback from "./WebPlayback";
import Login from "./Login";
import "./styles/SpotifyApp.css";

function SpotifyApp({ categoryName }) {
  const [token, setToken] = useState("");

  useEffect(() => {
    async function getToken() {
      const response = await fetch("/auth/token");
      const json = await response.json();
      setToken(json.access_token);
    }

    getToken();
  }, []);

  return (
    <>
      {token === "" ? (
        <Login />
      ) : (
        <WebPlayback categoryName={categoryName} token={token} />
      )}
    </>
  );
}

export default SpotifyApp;
