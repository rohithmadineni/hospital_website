const express = require('express');
const mongoose = require('mongoose');
const path = require('path');
const session = require('express-session');
const bcrypt = require('bcrypt'); // For password hashing
const { exec } = require('child_process');
const app = express();
const PORT = 3000;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files (CSS, JS)
app.use(express.static(path.join(__dirname, 'hospital')));

// Setup sessions
app.use(session({
    secret: 'hospital_secretkey', // Change this to a secure key in production
    resave: false,
    saveUninitialized: false,
    cookie: { secure: false } // Set to true if using HTTPS
}));

// Connect to MongoDB
mongoose.connect('mongodb://localhost:27017/hospitalDB')
    .then(() => console.log('Connected to MongoDB'))
    .catch(err => console.error('Error connecting to MongoDB:', err));

// User Schema
const userSchema = new mongoose.Schema({
    name: String,
    email: { type: String, unique: true },
    password: String,
    mobile: String,
    dob: Date
});

const User = mongoose.model('User', userSchema);

// Mongoose Schema for appointment
const appointmentSchema = new mongoose.Schema({
    name: String,
    number: String,
    doctor: String,
    date: String,
    user: { type: mongoose.Schema.Types.ObjectId, ref: 'User' } // Reference to user
});
const Appointment = mongoose.model('Appointment', appointmentSchema);

// Hash user passwords before saving
async function hashPassword(password) {
    return await bcrypt.hash(password, 10);
}

// Compare hashed passwords for login
async function comparePassword(password, hashedPassword) {
    return await bcrypt.compare(password, hashedPassword);
}

// Serve homepage
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '/hospital/homepage.html'));
});

// Serve login page
app.get('/login', (req, res) => {
    res.sendFile(path.join(__dirname, '/hospital/login.html'));
});

// Serve signup page
app.get('/signup', (req, res) => {
    res.sendFile(path.join(__dirname, '/hospital/signup.html'));
});

// Signup route
app.post('/signup', async (req, res) => {
    console.log('Signup request received:', req.body);
    try {
        const existingUser = await User.findOne({ email: req.body.email });
        if (existingUser) {
            console.log('Email already exists:', req.body.email);
            return res.status(400).json({ error: 'Email already exists. Please use a different email.' });
        }

        const hashedPassword = await hashPassword(req.body.password);
        const newUser = new User({
            name: req.body.name,
            email: req.body.email,
            password: hashedPassword,
            mobile: req.body.mobile,
            dob: req.body.dob
        });

        await newUser.save();

        // Automatically log in the user after signup
        req.session.user = newUser; // Set session user
        req.session.actions = []; // Initialize actions tracking
        console.log('Signup successful for:', req.body.email);
        res.status(201).json({ message: 'Signup successful! Redirecting to homepage...', redirect: true });
    } catch (err) {
        console.error('Error signing up:', err);
        res.status(500).json({ error: 'Error signing up. Please try again.' });
    }
});

// Login route
app.post('/login', async (req, res) => {
    console.log('Login request received:', req.body);
    const { username, password } = req.body;
    const user = await User.findOne({ $or: [{ email: username }, { mobile: username }] });

    if (!user) {
        console.log('User not found for username:', username);
        return res.status(401).json({ error: 'User not found' });
    }

    const isMatch = await comparePassword(password, user.password);

    if (isMatch) {
        req.session.user = user; // Set user session
        req.session.actions = [];
        console.log('Login successful for user:', user.email);
        res.json({ message: 'Login successful!', redirect: true });
    } else {
        console.log('Invalid credentials for user:', username);
        return res.status(401).json({ error: 'Invalid credentials' });
    }
});

// Middleware to check if logged in
function isAuthenticated(req, res, next) {
    if (req.session.user) {
        next();
    } else {
        res.redirect('/login');
    }
}

// Logout route
app.get('/logout', (req, res) => {
    req.session.destroy();
    res.redirect('/');
});

// Handle appointment submission (user must be logged in)
app.post('/appointment', isAuthenticated, async (req, res) => {
    const { name, number, doctor, date } = req.body;

    // Validate appointment data
    if (!name || !number || !doctor || !date) {
        return res.status(400).json({ error: 'All fields are required.' });
    }

    const newAppointment = new Appointment({
        name,
        number,
        doctor,
        date,
        user: req.session.user._id // Link to logged-in user
    });

    try {
        await newAppointment.save();
        // Track user action in session
        req.session.actions.push(`Booked appointment with Dr. ${doctor} on ${date}`);
        res.json({ message: 'Appointment booked successfully!' });
    } catch (err) {
        console.error('Error saving appointment:', err);
        res.status(500).json({ message: "Error saving appointment: " + err });
    }
});

// Route to check if user is logged in
app.get('/session-status', (req, res) => {
    if (req.session.user) {
        return res.json({ loggedIn: true, user: req.session.user });
    } else {
        return res.json({ loggedIn: false });
    }
});

// Start the brain tumor detection Flask server
exec('python hospital/braintumor/detection.py', (error, stdout, stderr) => {
    if (error) {
        console.error(`Error starting Flask server for brain tumor detection: ${error.message}`);
        return;
    }
    if (stderr) {
        console.error(`Flask server error for brain tumor detection: ${stderr}`);
        return;
    }
    console.log(`Brain Tumor Detection Flask server started: ${stdout}`);
});

// Route to handle brain tumor detection
app.get('/brain-tumor-detection', (req, res) => {
    res.redirect('http://localhost:5000'); // Redirect to brain tumor detection service on port 5000
});

// Start the pneumonia detection Flask server
exec('python hospital/pneumonia/detection.py', (error, stdout, stderr) => {
    if (error) {
        console.error(`Error starting Flask server for pneumonia detection: ${error.message}`);
        return;
    }
    if (stderr) {
        console.error(`Flask server error for pneumonia detection: ${stderr}`);
        return;
    }
    console.log(`Pneumonia Detection Flask server started: ${stdout}`);
});

// Route to handle pneumonia detection
app.get('/pneumonia-detection', (req, res) => {
    res.redirect('http://localhost:5001'); // Redirect to pneumonia detection service on port 5001
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
