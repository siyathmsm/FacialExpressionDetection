const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const sqlite3 = require('sqlite3').verbose();
const app = express();
const port = 3000;

app.use(cors());
app.use(express.json());
app.use(bodyParser.json());
app.use(express.static('public'));

// Set up the database
const db = new sqlite3.Database(':memory:');

db.serialize(() => {
    db.run("CREATE TABLE activities (id INTEGER PRIMARY KEY, activity TEXT)");
});

app.post('/api/activity', (req, res) => {
    const { activity } = req.body;
    db.run(`INSERT INTO activities (activity) VALUES (?)`, [activity], function(err) {
        if (err) {
            return res.status(500).json({ error: err.message });
        }
        res.json({ id: this.lastID, activity });
    });
});

// Dummy data for participants
const participants = [
    { id: 1, name: 'User 1' },
    { id: 2, name: 'User 2' },
    { id: 3, name: 'User 3' },
];

// Endpoint to get participants
app.get('/api/participants', (req, res) => {
    res.json(participants);
});

// Dummy data for analytics
let interested_count = 5;
let not_interest_count = 3;

app.get('/analytics_data', (req, res) => {
    res.json({
        interested_count: interested_count,
        not_interest_count: not_interest_count
    });
});

app.listen(port, () => {
    console.log(`Server running at http://127.0.0.1:${port}/`);
});