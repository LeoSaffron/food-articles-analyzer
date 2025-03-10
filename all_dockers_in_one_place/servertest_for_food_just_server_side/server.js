const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const mongoose = require('mongoose');
const path = require('path');
const connectDB = require('./db');  // Import the connectDB function
const app = express();


// Connect to MongoDB
connectDB();  // Establish the MongoDB connection




// Define the Recipe model (ensure this matches your MongoDB structure)
const Recipe = mongoose.model('Recipe', new mongoose.Schema({
  url_recipe: String,
  title: String,
  // Add other fields as needed
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

// New route for the list of URLs
app.get('/urls', async (req, res) => {
  try {
    const recipes = await Recipe.find();  // Get all recipes from the collection
    const urls = recipes.map(recipe => recipe.url_recipe);  // Extract the URLs
    res.render('urls', { urls: urls.join('\n') });  // Pass the URLs as plain text to a new EJS page
  } catch (error) {
    res.status(500).send('Error fetching data from the database');
  }
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