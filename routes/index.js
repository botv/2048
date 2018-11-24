const express = require('express');
const Agent = require('../models/agent');

const router = express.Router();
const agent = new Agent();

router.get('/', function(req, res) {
	res.render('index');
});

router.post('/move', function (req, res) {
	const state = JSON.parse(req.body.state);
	const action = agent.action(state);
	res.send(action.toString());
});

module.exports = router;