import random
import math
import numpy as np


class Game:

    def reset(self):
        self.board = [[0,0,0,0] for i in range(4)]
        firstRow = random.randint(0,len(self.board)-1)
        secondRow = random.randint(0,len(self.board)-1)
        firstCol = random.randint(0,len(self.board[0])-1)
        secondCol = random.randint(0,len(self.board[0])-1)
        while firstCol == secondCol:
            secondCol = random.randint(0,len(self.board[0])-1)
        self.board[firstRow][firstCol] = 2
        self.board[secondRow][secondCol] = 2

    def randomInsert(self):
        possibleCoords = []
        for i, row in enumerate(self.board):
            if 0 in row:
                for j, element in enumerate(row):
                    if element == 0:
                        possibleCoords.append([i,j])
        if len(possibleCoords) != 0:
            randomCoord = possibleCoords[random.randint(0,len(possibleCoords)) - 1]
            self.board[randomCoord[0]][randomCoord[1]] = 2


    def checkGameActive(self):
        for row in self.board:
            if 0 in row:
                return False
        for row in range(len(self.board)):
            for col in range(len(self.board[0])-1):
                if self.board[row][col] == self.board[row][col+1]:
                    return False
        for col in range(len(self.board[0])):
            for row in range(len(self.board)-1):
                if self.board[row][col] == self.board[row+1][col]:
                    return False
        return True


    def moveUp(self):
        for col in range(len(self.board[0])):
            colReal = []
            for r, row in enumerate(self.board):
                colReal.append(row[col])
            replace = self.merge(colReal)
            for i in range(len(self.board)):
                self.board[i][col] = replace[i]
        self.randomInsert()


    def moveDown(self):
        for col in range(len(self.board[0])):
            colReal = []
            for r, row in enumerate(self.board):
                colReal.append(row[col])
            replace = self.merge(colReal)
            replace = replace[::-1]
            for i in range(len(self.board)):
                self.board[i][col] = replace[i]
        self.randomInsert()


    def moveRight(self):
        for i, row in enumerate(self.board):
            replace = self.merge(row)
            replace = replace[::-1]
            self.board[i] = replace
        self.randomInsert()


    def moveLeft(self):
        for i, row in enumerate(self.board):
            replace = self.merge(row)
            self.board[i] = replace
        self.randomInsert()


    def merge(self, nums):
        prev = None
        store = []
        for next_ in nums:
            if not next_:
                continue
            if prev is None:
                prev = next_
            elif prev == next_:
                store.append(prev + next_)
                prev = None
            else:
                store.append(prev)
                prev = next_
        if prev is not None:
            store.append(prev)
        store.extend([0] * (len(nums) - len(store)))
        return store

    def getReward(self):
        logReward = 0
        for row in self.board:
            for element in row:
                if element != 0:
                    logReward += math.log(element, 2)
        return logReward


    def getState(self):
        state = []
        for row in self.board:
            for element in row:
                if element == 0:
                    state.append(0)
                else:
                    state.append(math.log(element, 2))
        return np.reshape(np.asarray(state), (1,len(self.board)**2))

    def step(self, action):
        if action == 0:
            self.moveUp()
        elif action == 1:
            self.moveDown()
        elif action == 2:
            self.moveRight()
        else:
            self.moveLeft()
        return self.getState(), self.getReward(), self.checkGameActive()
