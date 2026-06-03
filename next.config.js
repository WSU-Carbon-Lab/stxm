import "./src/env.js";

/** @type {import("next").NextConfig} */
const config = {
  transpilePackages: ["plotly.js", "react-plotly.js"],
};

export default config;
