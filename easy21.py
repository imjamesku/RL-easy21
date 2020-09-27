import random

# state contains:
# player sum, dealersum, terminated, reward
#


class Easy21:
    def __init__(self):
        pass

    def draw(self):
        num = random.randint(1, 10)
        color = random.randint(1, 3)
        # 3/1 red(negative) 3/2 black(positive)
        sign = -1 if color == 1 else 1
        return num * sign

    def startGame(self):
        """[summary]

        Returns:
            int: playerSum
            int: dealerSum
        """
        return random.randint(1, 10), random.randint(1, 10)

    def step(self, playerSum, dealerSum, action):
        """take a step in the env

        Args:
            playerSum (int): play's total sum
            dealerSum (int): dealer's total sum
            action (0 or 1): action player is taking

        Returns:
            playerSum (int): play's total sum
            dealerSum (int): dealer's total sum
            terminal (boolean): whether game has ended or not

        """
        assert action in [
            0, 1], "Expection action in [0, 1] but got {}".format(action)
        if action == 1:
            # hit
            playerSum += self.draw()
            if playerSum < 1 or playerSum > 21:
                return playerSum, dealerSum, True, -1
            return playerSum, dealerSum, False, 0
        else:
            # player sticks
            while dealerSum < playerSum and 1 <= dealerSum < 17:
                dealerSum += self.draw()
            if dealerSum < 1 or dealerSum > 21 or dealerSum < playerSum:
                return playerSum, dealerSum, True, 1
            elif dealerSum == playerSum:
                return playerSum, dealerSum, True, 0
            return playerSum, dealerSum, True, -1
