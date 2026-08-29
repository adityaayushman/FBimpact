/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // TensorFlow.js ships Node-targeted fallbacks that the browser bundle does not
  // need; without this the WebGL backend drags `fs` and `path` into the client
  // chunk and the build fails on Vercel.
  webpack: (config) => {
    config.resolve.fallback = { ...config.resolve.fallback, fs: false, path: false, crypto: false };
    return config;
  },
};

export default nextConfig;
