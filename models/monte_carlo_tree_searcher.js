const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const GameManager = require('../models/game_manager');
const StateActionNode = require('../models/state_action_node');
const Grid = require('../models/grid');
const Tile = require('../models/tile');

module.exports = class MonteCarloTreeSearcher {
	constructor(agent, depth = 10, traversals = 50) {
		this.depth = depth;
		this.traversals = traversals;

		this.agent = agent;

		// For the root, actions and probability do not matter; they should be null and should never be accessed
		this.root = new StateActionNode(this, null, null, null);

		// Store starting state for traversal
		this.state = null;

		// What to console.log during development
		this.logProcess = false;
		this.logMoves = true;
	}

	// Search from state
	// Return weights for actions (weight = numberOfVisits / traversals)
	// *traversals is also the sum of the number of visits to the first actions
	getCurrentWeights(state) {
		let traversals = 0;
		this.state = state;

		// Reset tree (reset root)
		this.root = new StateActionNode(this, null, null, null);

		do {
			// Traverse tree once
			this.traverse();
			traversals++;
		} while (traversals <= this.traversals);

		// construct weights
		let weights = [];
		for (let i = 0; i < this.root.children.length; i++) {
			weights.push(this.root.children[i].numberOfVisits / this.traversals)
		}

		if (this.logProcess) console.log("Constructed weights: " + weights);

		return weights;

	}

	// Single tree traversal
	traverse() {
		if (this.logProcess) console.log("Beginning traversal from state " + this.state + ": \n");

		// Start at root
		let currentNode = this.root;
		let depth = 0;

		// Keep track of previous nodes visited / path
		let states = [];

		// Game manager to simulate moves
		let simGameManager = new GameManager(4);
		simGameManager.setState(this.state);

		do {
			states.push(currentNode);

			if (this.logProcess) console.log("Evaluating node: root" + states.slice(1).map((a) => "->" + a.action).join(""));
			if (this.logMoves) {
				console.log("Previous state: ");
				if (depth !== 0) {
					GameManager.logState(currentNode.state);
					console.log("directions: " + simGameManager.directionsAvailable());
				}
				else console.log("none");
			}

			if (currentNode.numberOfVisits === 0) {
				// If node has never been visited before, rollout (choose randomly)
				// 0: up, 1: right, 2: down, 3: left
				if (this.logProcess) console.log("rollout");

				let actions = Array.apply(null, {length: 4}).map(Number.call, Number);
				let action = actions[Math.floor(Math.random()*actions.length)];

				// Set next node's state to current state; move current node's state (previous state) by current node's action
				// If depth is 0, next node's previous state is the current state (this.state)
				let currentState = null;
				if (depth === 0) {
					currentState = this.state;
				} else {
					simGameManager.move(currentNode.action);
					currentState = simGameManager.getState();
				}

				// Get probability of action from next node's previous state (current state) from agent
				let probability = this.agent.neuralNetPlaceholder(currentState)[action];

				// Do not add new node to children of current; Rollout is random and thus the children need not be recorded

				let nextNode = new StateActionNode(this, currentState, action, probability);
				if (this.logMoves) {
					console.log("Action: " + currentNode.action);
					console.log("Current state: ");
					GameManager.logState(currentState);
					console.log("");
				}

				currentNode.numberOfVisits++;
				depth++;

				currentNode = nextNode;
			} else if (currentNode.children.length < 4) {
				// If node has been visited and not all children are present, expand
				// add all 4 child nodes to children
				if (this.logProcess) console.log("construct children");

				for (let i = 0; i < 4; i++) {
					// 0: up, 1: right, 2: down, 3: left
					let action = i;

					// Set next node's state to next state; move current state by chosen action
					let currentState = null;
					if (depth === 0) {
						currentState = this.state;
					} else {
						// Copy simGameManager
						let copy = new GameManager(4);
						copy.setState(simGameManager.getState());

						// Move copy
						copy.move(currentNode.action);

						currentState = copy.getState();
					}

					// Get probability of action from (new) node's previous state from agent
					let probability = this.agent.neuralNetPlaceholder(currentState)[action];

					let nextNode = new StateActionNode(this, currentState, action, probability);

					currentNode.children.push(nextNode);
				}

				// select from children

				let nextNode = MonteCarloTreeSearcher.select(currentNode.children).node;

				// Simulate move with simGameManager
				if (depth === 0) {
					nextNode.state = this.state;
				} else {
					simGameManager.move(currentNode.action);
					nextNode.state = simGameManager.getState()
				}

				if (this.logMoves) {
					console.log("Action: " + currentNode.action);
					console.log("Current state: ");
					GameManager.logState(nextNode.state);
					console.log("");
				}

				currentNode.numberOfVisits++;
				depth++;

				currentNode = nextNode;
			} else {
				// If node has been visited and all children are present, select

				// Update child states if not root
				// This must be done because the current state could be different due to random apparitions
				if (depth > 0) {
					for (let i = 0; i < currentNode.children.length; i++) {
						currentNode.children[i].state = function () {
							// Copy simGameManager
							let copy = new GameManager(4);
							copy.setState(simGameManager.getState());

							// Move copy
							copy.move(currentNode.action);
							return copy.getState();
						}();
					}
				}

				let nextNode = MonteCarloTreeSearcher.select(currentNode.children).node;

				// Simulate move with simGameManager
				if (depth === 0) {
					nextNode.state = this.state;
				} else {
					simGameManager.move(currentNode.action);
					nextNode.state = simGameManager.getState()
				}

				if (this.logMoves) {
					console.log("Action: " + currentNode.action);
					console.log("Current state: ");
					GameManager.logState(nextNode.state);
					console.log("");
				}

				currentNode.numberOfVisits++;
				depth++;

				currentNode = nextNode;
			}

		} while (depth < this.depth && currentNode.reward(simGameManager) < 2048 && !simGameManager.isGameTerminated());

		// update values

		// Go backwards through states and update values
		// Value should be average reward of leaves
		// Ignore root (states[0])

		let reward = currentNode.reward(simGameManager);

		if (this.logProcess) console.log("Traversal complete at node root" + states.slice(1).map((a) => "->" + a.action).join(""));
		if (this.logProcess) console.log("Reward: " + reward + "\n");

		for (let i = states.length - 1; i > 0; i--) {
			let node = states[i];
			node.rewards.push(reward);

			node.value = node.rewards.reduce((a, b) => a + b, 0) / node.rewards.length;
		}

	}

	static select(nodeArray) {

		// Initialize next node to first element with a weight of -1, which will always be overwritten by the first node
		let nextNode = 0;
		let nextNodeWeight = -1;
		let i = 0;

		if (this.logProcess) console.log("Selecting action from: " + nodeArray.map((a) => a.action + ": " + a.value + (a.probability / (1 + a.numberOfVisits))));

		for (i; i < nodeArray.length; i++) {
			let node = nodeArray[i];

			let nodeWeight = node.value + (node.probability / (1 + node.numberOfVisits));

			if (isNaN(node.value)) {
				if (this.logProcess) console.log("node.rewards: " + node.rewards)
			}

			if (nodeWeight > nextNodeWeight) {
				nextNode = i;
				nextNodeWeight = nodeWeight;
			}
		}

		if (this.logProcess) console.log("Selected " + nextNode);

		return {
			index: nextNode,
			node: nodeArray[nextNode]
		};

	}

};