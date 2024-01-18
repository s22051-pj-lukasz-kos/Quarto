const { createProxyMiddleware } = require("http-proxy-middleware");

/**
 * Configures a proxy middleware for forwarding requests matching the "/auth/**" path to a specified target.
 * This middleware is intended to be used in a development environment.
 *
 * @param {Object} app - Express app instance.
 * @see {@link https://www.npmjs.com/package/http-proxy-middleware|http-proxy-middleware} for more details on middleware configuration.
 */
module.exports = function (app) {
  app.use(
    createProxyMiddleware(`/auth/**`, {
      target: "http://localhost:5000",
    })
  );
};
