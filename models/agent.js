module.exports = class Agent {
	static action(state) {
		return Math.floor(Math.random() * 4);
	}
};