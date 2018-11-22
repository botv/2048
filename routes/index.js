const express = require('express');
const Agent = require('../scripts/agent');
const router = express.Router();

router.get('/', function(req, res) {
	res.render('index', {multiplayer: true});
});

router.get('/computer', function(req, res) {
	res.render('index', {multiplayer: false})
});

router.post('/move', function (req, res) {
	const state = JSON.parse(req.body.state);
	const action = Agent.action(state);
	res.send(action.toString());
});

module.exports = router;
