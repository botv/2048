const express = require('express');
const router = express.Router();

router.get('/', function(req, res) {
	res.render('index');
});

router.post('/move', function (req, res) {
	const grid = JSON.parse(req.body.grid);
	console.log(grid);
	res.end();
});

module.exports = router;
