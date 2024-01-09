const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");

module.exports = {
  mode: "development",
  entry: ["./src/styles/main.scss", "./src/index.ts"],
  devServer: {
    static: {
      directory: path.join(__dirname, "build"),
      publicPath: "/",
    },
  },
  output: {
    filename: "bundle.js",
    path: path.resolve(__dirname, "build"),
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: "ts-loader",
        exclude: /node_modules/,
      },
      {
        test: /\.s[ac]ss$/i,
        use: [
          {
            loader: "style-loader",
          },
          { loader: "css-loader" },
          {
            loader: "sass-loader",
            options: {
              // Prefer Dart Sass
              implementation: require("sass"),

              // See https://github.com/webpack-contrib/sass-loader/issues/804
              webpackImporter: false,
              sassOptions: {
                includePaths: ["./node_modules"],
              },
            },
          },
        ],
        exclude: /node_modules/,
      },
    ],
  },
  plugins: [new HtmlWebpackPlugin({ template: "./src/index.html" })],
  resolve: {
    extensions: [".ts", ".js"],
  },
};
