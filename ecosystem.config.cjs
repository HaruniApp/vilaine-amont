module.exports = {
  apps: [{
    name: 'vigicrue-api',
    script: 'backend/src/index.js',
    env: {
      NODE_ENV: 'production',
      PORT: 3001,
    },
  }],
};
