module.exports = class Agent {
	action(state) {
		return Math.floor(Math.random() * 4);
	}
};