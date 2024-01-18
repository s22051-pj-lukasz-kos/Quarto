import React, { useState, useEffect } from "react";
import WebPlayback from "./WebPlayback";
import Login from "./Login";
import "./styles/SpotifyApp.css";

/**
 * React component representing the main Spotify application.
 *
 * @component
 * @param {Object} props - The props for the SpotifyApp component.
 * @param {string} props.categoryName - The name of the category to be displayed.
 * @returns {JSX.Element} - The rendered SpotifyApp component.
 */
function SpotifyApp({ categoryName }) {
  /**
   * State to store the Spotify access token.
   *
   * @type {string}
   */
  const [token, setToken] = useState("");

  /**
   * useEffect hook to fetch the Spotify access token when the component mounts.
   *
   * @function
   * @async
   * @listens useEffect
   */
  useEffect(() => {
    /**
     * Fetches the Spotify access token from the server and updates the state.
     *
     * @function
     * @async
     */
    async function getToken() {
      const response = await fetch("/auth/token");
      const json = await response.json();
      setToken(json.access_token);
    }

    getToken();
  }, []);

  /**
   * Renders the SpotifyApp component.
   *
   * If the access token is empty, it renders the Login component.
   * Otherwise, it renders the WebPlayback component with the provided category name and token.
   *
   * @returns {JSX.Element} - The rendered component.
   */
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
