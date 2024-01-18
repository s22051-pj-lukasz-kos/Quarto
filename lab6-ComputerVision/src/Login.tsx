import React from "react";



/**
 * Login component renders a button for logging in with Spotify.
 *
 * @component
 * @returns {JSX.Element} - The rendered Login component.
 */
function Login() {
  return (
    <div className="App">
      <header className="App-header">
        <a className="btn-spotify" href="/auth/login">
          Login with Spotify
        </a>
      </header>
    </div>
  );
}

export default Login;
