const express = require("express");
const dotenv = require("dotenv");
const request = require("request");

// Port on which the Express server will run
const port = 5000;

// Global variable to store the Spotify access token
global.access_token = "";

// Load environment variables from a .env file
dotenv.config();

// Retrieve Spotify client ID and client secret from environment variables
var spotify_client_id = process.env.SPOTIFY_CLIENT_ID;
var spotify_client_secret = process.env.SPOTIFY_CLIENT_SECRET;

var spotify_redirect_uri = "http://localhost:3000/auth/callback";


/**
 * Generates a random string of the specified length.
 *
 * @param {number} length - The length of the random string.
 * @returns {string} - The generated random string.
 */
var generateRandomString = function (length) {
  var text = "";
  var possible =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

  for (var i = 0; i < length; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }
  return text;
};

var app = express();

/**
 * Handles the authentication process by redirecting the user to the Spotify authorization page.
 */
app.get("/auth/login", (req, res) => {
  var scope = "streaming user-read-email user-read-private";
  var state = generateRandomString(16);

  var auth_query_parameters = new URLSearchParams({
    response_type: "code",
    client_id: spotify_client_id,
    scope: scope,
    redirect_uri: spotify_redirect_uri,
    state: state,
  });

  res.redirect(
    "https://accounts.spotify.com/authorize/?" +
      auth_query_parameters.toString()
  );
});

/**
 * Handles the Spotify callback after user authorization, retrieves the access token,
 * and stores it in the global variable.
 */
app.get("/auth/callback", (req, res) => {
  var code = req.query.code;

  var authOptions = {
    url: "https://accounts.spotify.com/api/token",
    form: {
      code: code,
      redirect_uri: spotify_redirect_uri,
      grant_type: "authorization_code",
    },
    headers: {
      Authorization:
        "Basic " +
        Buffer.from(spotify_client_id + ":" + spotify_client_secret).toString(
          "base64"
        ),
      "Content-Type": "application/x-www-form-urlencoded",
    },
    json: true,
  };

  request.post(authOptions, function (error, response, body) {
    if (!error && response.statusCode === 200) {
      global.access_token = body.access_token;
      res.redirect("/");
    }
  });
});

/**
 * Endpoint to retrieve the stored Spotify access token.
 */
app.get("/auth/token", (req, res) => {
  res.json({
    access_token: global.access_token,
  });
});

app.listen(port, () => {
  console.log(`Listening at http://localhost:${port}`);
});
