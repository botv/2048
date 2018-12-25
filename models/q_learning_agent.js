const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const GameManager = require('../models/game_manager');

module.exports = class QLearningAgent {
	constructor(learningRate = 0.001) {
		this.data = {};
		this.learningRate = learningRate;
	}

	// Train agent with certain amount of games
	train(games) {
		const gameManager = new GameManager(4);

		for (let i = 1; i <= games; i++) {
			let actions = {};

			do {
				// Initialize variables
				const stateString = JSON.stringify(gameManager.getState());

				// Get current weights for state
				let weights = this.getCurrentWeights(stateString);

				// Perform action
				const action = tf.multinomial(tf.log(weights), 1).dataSync()[0];
				actions[stateString] = action;
				gameManager.move(action);
			} while (!gameManager.isGameTerminated());

			// Get cumulative reward
			const cumulativeReward = gameManager.getCumulativeReward();
			const adjustedCumulativeReward = cumulativeReward * this.learningRate;

			// Update weights
			for (let stateString in actions) {
				const action = actions[stateString];

				// Get current weights for state
				let weights = this.getCurrentWeights(stateString);
				weights[action] += adjustedCumulativeReward;

				// Update weights
				this.data[stateString] = tf.tensor1d(weights).softmax().dataSync();
			}

			console.log(`(${i}) Cumulative reward: ${cumulativeReward}`);
			gameManager.restart();
		}
	}

	// Get current weights for state
	getCurrentWeights(stateString) {
		let weights;
		if (!this.data[stateString]) weights = tf.ones([4]).softmax().dataSync();
		else weights = this.data[stateString];

		return weights;
	}

	// Get state string rotated 0, 90, 180 and 270 degrees
	getRotatedStateStrings(stateString) {
		// Convert stateString to a tensor
		const state = tf.tensor2d(JSON.parse(stateString), [4, 4]);

		// Rotate matrix by some multiple of 90 degrees
		function rotate(matrix, degrees) {
			for (let i = 0; i < degrees / 90; i++) {
				matrix = matrix.reverse(1).transpose();
			}

			return matrix;
		}

		// Get rotated matrices
		const r0 = rotate(state, 0);
		const r90 = rotate(state, 90);
		const r180 = rotate(state, 180);
		const r270 = rotate(state, 270);

		// Change tensor into string
		function tensorToString(tensor) {
			return JSON.stringify(Array.from(tensor.dataSync()).flat());
		}

		// Return rotated state strings
		return [
			tensorToString(r0),
			tensorToString(r90),
			tensorToString(r180),
			tensorToString(r270)
		];
	}

	// Summarize training by logging states that were adjusted more than once
	summarize() {
		for (let stateString in this.data) {
			if (this.data[stateString].length < Array.from(new Set(this.data[stateString])).length + 2) {
				console.log(stateString + ': ' + this.data[stateString]);
			}
		}
	}
};