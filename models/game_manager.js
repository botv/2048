const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const Tile = require('./tile');
const Grid = require('./grid');
const Agent = require('./agent');

module.exports = class GameManager {
	constructor(size) {
		this.size = size; // Size of the grid
		this.startTiles = 2;

		this.setup();
	}

	static directionString(direction) {
		switch (direction) {
			case 0:
				return 'up';
			case 1:
				return 'right';
			case 2:
				return 'down';
			case 3:
				return 'left';
		}
	};

	static positionsEqual(first, second) {
		return first.x === second.x && first.y === second.y;
	}

	// Get the vector representing the chosen direction
	static getVector(direction) {
		// Vectors representing tile movement
		const map = {
			0: {x: 0, y: -1}, // Up
			1: {x: 1, y: 0}, // Right
			2: {x: 0, y: 1}, // Down
			3: {x: -1, y: 0} // Left
		};

		return map[direction];
	}

	// Play specified number of games
	run(games) {
		for (let i = 0; i < games; i++) {
			do {
				const action = this.agent.action(this.serialize());
				this.move(action);
			} while (!this.isGameTerminated());

			console.log(this.serialize());
			console.log(this.getState());
			console.log(this.getCumulativeReward());

			this.restart();
		}
	};

	// Restart the game
	restart() {
		this.setup();
	}

	// Return true if the game is lost, or has won and the user hasn't kept playing
	isGameTerminated() {
		return this.over || this.won;
	}

	// Set up the game
	setup() {
		this.grid = new Grid(this.size);
		this.agent = new Agent();
		this.score = 0;
		this.over = false;
		this.won = false;

		// Add the initial tiles
		this.addStartTiles();
	}

	// Set up the initial tiles to start the game with
	addStartTiles() {
		for (let i = 0; i < this.startTiles; i++) {
			this.addRandomTile();
		}
	}

	// Adds a tile in a random position
	addRandomTile() {
		if (this.grid.cellsAvailable()) {
			const value = Math.random() < 0.9 ? 2 : 4;
			const tile = new Tile(this.grid.randomAvailableCell(), value);

			this.grid.insertTile(tile);
		}
	}

	// Represent the current game as an object
	serialize() {
		return {
			grid: this.grid.serialize(),
			score: this.score,
			over: this.over,
			won: this.won
		};
	}

	// Save all tile positions and remove merger info
	prepareTiles() {
		this.grid.eachCell(function (x, y, tile) {
			if (tile) {
				tile.mergedFrom = null;
				tile.savePosition();
			}
		});
	}

	// Move a tile and its representation
	moveTile(tile, cell) {
		this.grid.cells[tile.x][tile.y] = null;
		this.grid.cells[cell.x][cell.y] = tile;
		tile.updatePosition(cell);
	}

	// Move tiles on the grid in the specified direction
	move(direction) {
		// 0: up, 1: right, 2: down, 3: left
		const self = this;

		if (this.isGameTerminated()) return; // Don't do anything if the game's over

		let cell, tile;

		const vector = GameManager.getVector(direction);
		const traversals = this.buildTraversals(vector);
		let moved = false;

		// Save the current tile positions and remove merger information
		this.prepareTiles();

		// Traverse the grid in the right direction and move tiles
		traversals.x.forEach(function (x) {
			traversals.y.forEach(function (y) {
				cell = {x: x, y: y};
				tile = self.grid.cellContent(cell);

				if (tile) {
					const positions = self.findFarthestPosition(cell, vector);
					const next = self.grid.cellContent(positions.next);

					// Only one merger per row traversal?
					if (next && next.value === tile.value && !next.mergedFrom) {
						const merged = new Tile(positions.next, tile.value * 2);
						merged.mergedFrom = [tile, next];

						self.grid.insertTile(merged);
						self.grid.removeTile(tile);

						// Converge the two tiles' positions
						tile.updatePosition(positions.next);

						// Update the score
						self.score += merged.value;

						// The mighty 2048 tile
						if (merged.value === 2048) self.won = true;
					} else {
						self.moveTile(tile, positions.farthest);
					}

					if (!GameManager.positionsEqual(cell, tile)) {
						moved = true; // The tile moved from its original cell!
					}
				}
			});
		});

		if (moved) {
			this.addRandomTile();

			if (!this.movesAvailable()) {
				this.over = true; // Game over!
			}
		}
	}

	// Build a list of positions to traverse in the right order
	buildTraversals(vector) {
		const traversals = {x: [], y: []};

		for (let pos = 0; pos < this.size; pos++) {
			traversals.x.push(pos);
			traversals.y.push(pos);
		}

		// Always traverse from the farthest cell in the chosen direction
		if (vector.x === 1) traversals.x = traversals.x.reverse();
		if (vector.y === 1) traversals.y = traversals.y.reverse();

		return traversals;
	}

	findFarthestPosition(cell, vector) {
		let previous;

		// Progress towards the vector direction until an obstacle is found
		do {
			previous = cell;
			cell = {x: previous.x + vector.x, y: previous.y + vector.y};
		} while (this.grid.withinBounds(cell) && this.grid.cellAvailable(cell));

		return {
			farthest: previous,
			next: cell // Used to check if a merge is required
		};
	}

	movesAvailable() {
		return this.grid.cellsAvailable() || this.tileMatchesAvailable();
	}

	tileMatchesAvailable() {
		const self = this;

		let tile;

		for (let x = 0; x < this.size; x++) {
			for (let y = 0; y < this.size; y++) {
				tile = this.grid.cellContent({x: x, y: y});

				if (tile) {
					for (let direction = 0; direction < 4; direction++) {
						const vector = GameManager.getVector(direction);
						const cell = {x: x + vector.x, y: y + vector.y};

						const other = self.grid.cellContent(cell);

						if (other && other.value === tile.value) {
							return true; // These two tiles can be merged
						}
					}
				}
			}
		}

		return false;
	};

	getCumulativeReward() {
		return this.grid.sum();
	}

	getState() {
		return this.grid.flatten().map(x => x == null ? 0 : Math.log2(x.value));
	}

	train(games) {
		let data = {};

		for (let i = 0; i < games; i++) {
			do {
				// initialize variables
				const stateString = JSON.stringify(this.getState());
				const cumulativeReward1 = this.getCumulativeReward();

				// get current weights for state
				let weights;
				if (!data[stateString]) weights = tf.ones([4]).softmax().dataSync();
				else weights = data[stateString];

				// perform action
				const action = tf.multinomial(tf.log(weights), 1).dataSync()[0];
				this.move(action);

				// get reward of action
				const cumulativeReward2 = this.getCumulativeReward();
				const reward = cumulativeReward2 - cumulativeReward1;

				// update weights
				weights[action] += reward;
				data[stateString] = tf.tensor1d(weights).softmax().dataSync();
			} while (!this.isGameTerminated());

			console.log('Cumulative reward: ' + this.getCumulativeReward());
			this.restart();
		}

		console.log(data);
	}
};