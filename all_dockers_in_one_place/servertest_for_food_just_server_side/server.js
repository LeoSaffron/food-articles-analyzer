const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const mongoose = require('mongoose');
const path = require('path');
const connectDB = require('./db');  // Import the connectDB function
const app = express();

// Connect to MongoDB
connectDB();

// Define the Recipe model
const Recipe = mongoose.model('Recipe', new mongoose.Schema({
  url_recipe: String,
  title: String,
}), 'recipes_tasty_co');

// Set EJS as the templating engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Middleware to parse URL-encoded bodies (from form submissions)
app.use(bodyParser.urlencoded({ extended: false }));

// Serve static files (like CSS) from the "public" directory
app.use(express.static('public'));

// Home page with the form
app.get('/', (req, res) => {
  res.render('index');
});

// Route to show all available recipe URLs
app.get('/urls', async (req, res) => {
  try {
    const recipes = await Recipe.find();
    const urls = recipes.map(recipe => recipe.url_recipe);
    res.render('urls', { urls: urls.join('\n') });
  } catch (error) {
    res.status(500).send('Error fetching data from the database');
  }
});

// Select which backend to use (New Agent or Old Backend)
const USE_NEW_AGENT = process.env.USE_NEW_AGENT === 'true';
const NEW_AGENT_URL = "http://host.docker.internal:8002/check_recipe";
const OLD_BACKEND_URL = "http://host.docker.internal:8086/analyze";

// Form submission: Call either the new agent or the old backend
app.post('/analyze', async (req, res) => {
  const recipeUrl = req.body.url;
  if (!recipeUrl) {
    return res.send("Missing recipe URL");
  }

  try {
    // Backend should make the request, not the browser
    const apiUrl = USE_NEW_AGENT ? NEW_AGENT_URL : OLD_BACKEND_URL;

    console.log(`Proxying request to: ${apiUrl} for URL: ${recipeUrl}`);

    const response = await axios.get(apiUrl, { params: { url: recipeUrl } });
    res.json(response.data);  // Send data back to the frontend
  } catch (err) {
    console.error("Error calling API:", err);
    res.status(500).send("Error contacting API: " + err.message);
  }
});


// Start the Express server
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Express server listening on port ${port}`);
  console.log(`Using ${USE_NEW_AGENT ? "New Agent (8002)" : "Old Backend (8086)"}`);
});