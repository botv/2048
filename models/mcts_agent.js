const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const GameManager = require('../models/game_manager');
const MonteCarloTreeSearcher = require('../models/monte_carlo_tree_searcher');

module.exports = class MCTSAgent {
	constructor(learningRate = 0.001, depth = 10, traversals = 150) {
		this.data = {};
		this.learningRate = learningRate;
		this.mcts = new MonteCarloTreeSearcher(this, depth, traversals);
	}

	// train agent with certain amount of games
	train(games) {
		const gameManager = new GameManager(4);

		for (let i = 1; i <= games; i++) {
			let actions = {};

			do {
				// initialize variables
				const stateString = JSON.stringify(gameManager.getState());

				// get action
				let weights = this.mcts.getCurrentWeights(gameManager.getState());
				console.log(weights);
				const action = weights.indexOf(Math.max(...weights));
				actions[stateString] = action;
				gameManager.move(action);
			} while (!gameManager.isGameTerminated());

			// get cumulative reward
			const cumulativeReward = gameManager.getCumulativeReward();
			const adjustedCumulativeReward = cumulativeReward * this.learningRate;

			// update weights
			for (let stateString in actions) {
				const action = actions[stateString];

				// get current weights for state
				let weights = this.getCurrentWeights(stateString);
				weights[action] += adjustedCumulativeReward;
				this.data[stateString] = tf.tensor1d(weights).softmax().dataSync();
			}

			console.log(`(${i}) Cumulative reward: ${cumulativeReward}`);
			gameManager.restart();
		}
	}

	// get current weights for state
	getCurrentWeights(stateString) {
		let weights;
		if (!this.data[stateString]) weights = tf.ones([4]).softmax().dataSync();
		else weights = this.data[stateString];

		return weights;
	}

	neuralNetPlaceholder(state) {
		// Use q learning weights until neural net is complete, then use output of NN

		// initialize variables
		const stateString = JSON.stringify(state);

		// get current weights for state
		let weights = this.getCurrentWeights(stateString);

		// return weights
		return weights;
	}

	// get state string rotated 0, 90, 180 and 270 degrees.
	getRotatedStateString(stateString) {
		// convert stateString to a tensor
		const state = tf.tensor2d(JSON.parse(stateString), [4, 4]);

		// define rotation matrices (these do not work)
		const r0 = tf.tensor2d([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
		const r90 = tf.tensor2d([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]);
		const r180 = tf.tensor2d([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]);
		const r270 = tf.tensor2d([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]);

		// change tensor into string
		function tensorToString(tensor) {
			return JSON.stringify(Array.from(tensor.dataSync()).flat());
		}

		// return rotated state strings
		return [
			tensorToString(tf.mul(state, r0)),
			tensorToString(tf.mul(state, r90)),
			tensorToString(tf.mul(state, r180)),
			tensorToString(tf.mul(state, r270))
		];
	}

	// summarize training by logging states that were adjusted more than once
	summarize() {
		for (let stateString in this.data) {
			if (this.data[stateString].length < Array.from(new Set(this.data[stateString])).length + 2) {
				console.log(stateString + ': ' + this.data[stateString]);
			}
		}
	}
};