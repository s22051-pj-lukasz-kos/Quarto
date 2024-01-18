import React, { useState, useEffect } from "react";


/**
 * Default track object structure.
 *
 * @typedef {Object} Track
 * @property {string} name - The name of the track.
 * @property {Object} album - The album details of the track.
 * @property {Object[]} album.images - An array of images associated with the album.
 * @property {string} album.images.url - The URL of the album cover image.
 * @property {Object[]} artists - An array of artists associated with the track.
 * @property {string} artists.name - The name of the artist.
 */
const track = {
  name: "",
  album: {
    images: [{ url: "" }],
  },
  artists: [{ name: "" }],
};

/**
 * WebPlayback component handles Spotify Web Playback SDK integration and controls.
 *
 * @component
 * @param {Object} props - React props for the component.
 * @param {string} props.token - The Spotify access token for authentication.
 * @param {string} props.categoryName - The name of the gesture category.
 * @returns {JSX.Element} - The rendered WebPlayback component.
 */
function WebPlayback({ token, categoryName }) {
  /**
   * State to track whether the playback is paused.
   *
   * @type {boolean}
   */
  const [is_paused, setPaused] = useState(false);

  /**
   * State to track whether the playback is active.
   *
   * @type {boolean}
   */
  const [is_active, setActive] = useState(false);
  const [player, setPlayer] = useState(undefined);
  const [current_track, setTrack] = useState(track);
  const [oldCategoryName, setOldCategoryName] = useState<String>("");

  useEffect(() => {
    const script = document.createElement("script");
    script.src = "https://sdk.scdn.co/spotify-player.js";
    script.async = true;

    document.body.appendChild(script);

    window.onSpotifyWebPlaybackSDKReady = () => {
      const player = new window.Spotify.Player({
        name: "HandGesture Spotify",
        getOAuthToken: (cb) => {
          cb(token);
        },
        volume: 0.5,
      });

      setPlayer(player);

      // when the SDK is connected and ready to stream content
      player.addListener("ready", ({ device_id }) => {
        console.log("Ready with Device ID", device_id);
      });

      // in case the connection is broken.
      player.addListener("not_ready", ({ device_id }) => {
        console.log("Device ID has gone offline", device_id);
      });

      // when the state of the local playback has changed (i.e., change of track)
      player.addListener("player_state_changed", (state) => {
        if (!state) {
          return;
        }

        setTrack(state.track_window.current_track);
        setPaused(state.paused);

        player.getCurrentState().then((state) => {
          !state ? setActive(false) : setActive(true);
        });
      });

      // Connect our Web Playback SDK instance to Spotify with the credentials provided during initialization.
      player.connect();
    };
  }, [token]);

  /**
   * useEffect hook to handle gesture-based controls for playback.
   */
  useEffect(() => {
    switch (categoryName) {
      case "Pointing_Up":
        if (categoryName !== oldCategoryName) {
          player.resume();
          console.log("Pointing_Up");
        }
        break;
      case "Closed_Fist":
        if (categoryName !== oldCategoryName) {
          player.pause();
          console.log("Closed_Fist");
        }
        break;
      case "Thumb_Up":
        if (categoryName !== oldCategoryName) {
          player.nextTrack();
          console.log("Thumb_Up");
        }
        break;
      case "Thumb_Down":
        if (categoryName !== oldCategoryName) {
          player.previousTrack();
          console.log("Thumb_Down");
        }
        break;
      default:
        console.log("Unrecognized gesture");
    }
    setOldCategoryName(categoryName);
  }, [categoryName, player]);

  if (!is_active) {
    return (
      <>
        <div className="container">
          <div className="main-wrapper">
            <b>
              {" "}
              Instance not active. In Spotify app (like on your phone) tap
              device and choose "HandGesture Spotify"{" "}
            </b>
          </div>
        </div>
      </>
    );
  } else {
    return (
      <>
        <div className="container">
          <div className="main-wrapper">
            <img
              src={current_track.album.images[0].url}
              className="now-playing__cover"
              alt=""
            />

            <div className="now-playing__side">
              <div className="now-playing__name">{current_track.name}</div>
              <div className="now-playing__artist">
                {current_track.artists[0].name}
              </div>

              <button
                className="btn-spotify"
                onClick={() => {
                  player.previousTrack();
                }}
              >
                &lt;&lt;
              </button>

              <button
                className="btn-spotify"
                onClick={() => {
                  player.togglePlay();
                }}
              >
                {is_paused ? "PLAY" : "PAUSE"}
              </button>

              <button
                className="btn-spotify"
                onClick={() => {
                  player.nextTrack();
                }}
              >
                &gt;&gt;
              </button>
            </div>
          </div>
        </div>
      </>
    );
  }
}

export default WebPlayback;
