const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const app = express();

// Set EJS as the templating engine
app.set('view engine', 'ejs');

// Middleware to parse URL-encoded bodies (from form submissions)
app.use(bodyParser.urlencoded({ extended: false }));

// Serve static files (like CSS) from the "public" directory
app.use(express.static('public'));

// Home page with the form
app.get('/', (req, res) => {
  res.render('index');
});

// Form submission: server-side call to the Flask API
app.post('/analyze', async (req, res) => {
  const recipeUrl = req.body.url;
  if (!recipeUrl) {
    return res.send("Missing recipe URL");
  }

  try {
    // Replace with your Flask API endpoint
    //const response = await axios.get('http://localhost:8086/analyze', {
    //  params: { url: recipeUrl }
    //});
	const response = await axios.get('http://host.docker.internal:8086/analyze', { params: { url: recipeUrl } });
    const analysis = response.data;
    res.render('results', { analysis });
  } catch (err) {
    console.error("Error calling Flask API:", err);
    res.send("Error contacting Flask API: " + err.message);
  }
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Express server listening on port ${port}`);
});