function Html_actuator() {
	this.tileContainer = document.querySelector('.tile-container');
	this.scoreContainer = document.querySelector('.score-container');
	this.bestContainer = document.querySelector('.best-container');
	this.messageContainer = document.querySelector('.game-message');

	this.score = 0;
}

Html_actuator.prototype.actuate = function (grid, metadata) {
	const self = this;

	window.requestAnimationFrame(function () {
		self.clearContainer(self.tileContainer);

		grid.cells.forEach(function (column) {
			column.forEach(function (cell) {
				if (cell) {
					self.addTile(cell);
				}
			});
		});

		self.updateScore(metadata.score);
		self.updateBestScore(metadata.bestScore);

		if (metadata.terminated) {
			if (metadata.over) {
				self.message(false); // You lose
			} else if (metadata.won) {
				self.message(true); // You win!
			}
		}

	});
};

// Continues the game (both restart and keep playing)
Html_actuator.prototype.continueGame = function () {
	this.clearMessage();
};

Html_actuator.prototype.clearContainer = function (container) {
	while (container.firstChild) {
		container.removeChild(container.firstChild);
	}
};

Html_actuator.prototype.addTile = function (tile) {
	const self = this;

	const wrapper = document.createElement('div');
	const inner = document.createElement('div');
	const position = tile.previousPosition || {x: tile.x, y: tile.y};
	const positionClass = this.positionClass(position);

	// We can't use classlist because it somehow glitches when replacing classes
	const classes = ['tile', 'tile-' + tile.value, positionClass];

	if (tile.value > 2048) classes.push('tile-super');

	this.applyClasses(wrapper, classes);

	inner.classList.add('tile-inner');
	inner.textContent = tile.value;

	if (tile.previousPosition) {
		// Make sure that the tile gets rendered in the previous position first
		window.requestAnimationFrame(function () {
			classes[2] = self.positionClass({x: tile.x, y: tile.y});
			self.applyClasses(wrapper, classes); // Update the position
		});
	} else if (tile.mergedFrom) {
		classes.push('tile-merged');
		this.applyClasses(wrapper, classes);

		// Render the tiles that merged
		tile.mergedFrom.forEach(function (merged) {
			self.addTile(merged);
		});
	} else {
		classes.push('tile-new');
		this.applyClasses(wrapper, classes);
	}

	// Add the inner part of the tile to the wrapper
	wrapper.appendChild(inner);

	// Put the tile on the board
	this.tileContainer.appendChild(wrapper);
};

Html_actuator.prototype.applyClasses = function (element, classes) {
	element.setAttribute('class', classes.join(' '));
};

Html_actuator.prototype.normalizePosition = function (position) {
	return {x: position.x + 1, y: position.y + 1};
};

Html_actuator.prototype.positionClass = function (position) {
	position = this.normalizePosition(position);
	return 'tile-position-' + position.x + '-' + position.y;
};

Html_actuator.prototype.updateScore = function (score) {
	this.clearContainer(this.scoreContainer);

	const difference = score - this.score;
	this.score = score;

	this.scoreContainer.textContent = this.score;

	if (difference > 0) {
		const addition = document.createElement('div');
		addition.classList.add('score-addition');
		addition.textContent = '+' + difference;

		this.scoreContainer.appendChild(addition);
	}
};

Html_actuator.prototype.updateBestScore = function (bestScore) {
	this.bestContainer.textContent = bestScore;
};

Html_actuator.prototype.message = function (won) {
	const type = won ? 'game-won' : 'game-over';
	const message = won ? 'You win!' : 'Game over!';

	this.messageContainer.classList.add(type);
	this.messageContainer.getElementsByTagName('p')[0].textContent = message;
};

Html_actuator.prototype.clearMessage = function () {
	// IE only takes one value to remove at a time.
	this.messageContainer.classList.remove('game-won');
	this.messageContainer.classList.remove('game-over');
};
