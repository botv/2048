module.exports = class Tile {
	constructor(position, value) {
		this.x = position.x;
		this.y = position.y;
		this.value = (value == null) ?  2 : value;

		this.previousPosition = null;
		this.mergedFrom = null; // Tracks tiles that merged together
	}

	savePosition() {
		this.previousPosition = {x: this.x, y: this.y};
	};

	updatePosition(position) {
		this.x = position.x;
		this.y = position.y;
	};

	serialize() {
		return {
			position: {
				x: this.x,
				y: this.y
			},
			value: this.value
		};
	};
};