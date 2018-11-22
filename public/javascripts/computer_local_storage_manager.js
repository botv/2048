window.fakeStorage = {
	_data: {},

	setItem: function (id, val) {
		return this._data[id] = String(val);
	},

	getItem: function (id) {
		return this._data.hasOwnProperty(id) ? this._data[id] : undefined;
	},

	removeItem: function (id) {
		return delete this._data[id];
	},

	clear: function () {
		return this._data = {};
	}
};

function ComputerLocalStorageManager() {
	this.bestScoreKey = 'computerBestScore';
	this.gameStateKey = 'computerGameState';

	const supported = this.localStorageSupported();
	this.storage = supported ? window.localStorage : window.fakeStorage;
}

ComputerLocalStorageManager.prototype.localStorageSupported = function () {
	const testKey = 'computerTest';

	try {
		const storage = window.localStorage;
		storage.setItem(testKey, '1');
		storage.removeItem(testKey);
		return true;
	} catch (error) {
		return false;
	}
};

// Best score getters/setters
ComputerLocalStorageManager.prototype.getBestScore = function () {
	return this.storage.getItem(this.bestScoreKey) || 0;
};

ComputerLocalStorageManager.prototype.setBestScore = function (score) {
	this.storage.setItem(this.bestScoreKey, score);
};

// Game state getters/setters and clearing
ComputerLocalStorageManager.prototype.getGameState = function () {
	const stateJSON = this.storage.getItem(this.gameStateKey);
	return stateJSON ? JSON.parse(stateJSON) : null;
};

ComputerLocalStorageManager.prototype.setGameState = function (gameState) {
	this.storage.setItem(this.gameStateKey, JSON.stringify(gameState));
};

ComputerLocalStorageManager.prototype.clearGameState = function () {
	this.storage.removeItem(this.gameStateKey);
};
