const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const GameManager = require('../models/game_manager');
const MonteCarloTreeSearcher = require('../models/monte_carlo_tree_searcher');

module.exports = class StateActionNode {
	constructor(mcts, state, action, probability) {
		this.state = state;
		this.action = action;

		this.numberOfVisits = 0;
		this.value = 0;
		this.probability = probability;

		this.children = [];

		// keep track of rewards for updating value
		this.rewards = [];
	}

	reward(gameManager) {
		// Copy gameManager
		// Copy simGameManager
		let copy = new GameManager(4);
		copy.setState(gameManager.getState());

		// Move copy
		copy.move(this.action);

		// If root, set state to 0
		return this.state ? Math.pow(2, Math.max(...copy.getState())) / 2048 : 0;
	}

};