const express = require('express');

const router = express.Router();

router.get('/', function(req, res) {
	res.render('index');
});

router.post('/move', function (req, res) {
	const action = Math.floor(Math.random() * 4);
	res.send(action.toString());
});

module.exports = router;