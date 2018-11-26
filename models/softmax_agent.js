const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const GameManager = require('../models/game_manager');

module.exports = class SoftmaxAgent {
	constructor() {
		this.data = {};
	}

	train(games) {
		const gameManager = new GameManager(4);

		for (let i = 0; i < games; i++) {
			do {
				// initialize variables
				const stateString = JSON.stringify(gameManager.getState());
				const cumulativeReward1 = gameManager.getCumulativeReward();

				// get current weights for state
				let weights;
				if (!this.data[stateString]) weights = tf.ones([4]).softmax().dataSync();
				else weights = this.data[stateString];

				// perform action
				const action = tf.multinomial(tf.log(weights), 1).dataSync()[0];
				gameManager.move(action);

				// get reward of action
				const cumulativeReward2 = gameManager.getCumulativeReward();
				const reward = cumulativeReward2 - cumulativeReward1;

				// update weights
				weights[action] += reward;
				this.data[stateString] = tf.tensor1d(weights).softmax().dataSync();
			} while (!gameManager.isGameTerminated());

			console.log('Cumulative reward: ' + gameManager.getCumulativeReward());
			gameManager.restart();
		}
	}
};