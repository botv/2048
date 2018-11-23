const express = require('express');
const Agent = require('../scripts/agent');
const router = express.Router();

router.get('/', function(req, res) {
	res.render('index');
});

router.post('/move', function (req, res) {
	const grid = JSON.parse(req.body.grid);
	const action = Agent.action(grid);
	res.send(action.toString());
});

module.exports = router;