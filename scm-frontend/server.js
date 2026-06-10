const express = require("express");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;

// Works whether files are flat or nested — always resolves relative to this file
const publicDir = path.join(__dirname, "public");

app.use(express.static(publicDir));

app.get("*", (req, res) => {
  res.sendFile(path.join(publicDir, "index.html"));
});

app.listen(PORT, () => {
  console.log(`SCM Frontend running on port ${PORT}`);
  console.log(`Serving static files from: ${publicDir}`);
});
