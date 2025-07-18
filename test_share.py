# test_share.py
from human import Human, House
from caracteristics import TrustSystem

# create two adjacent houses and two humans
h1_home = House(0, 0, (255,0,0))
h2_home = House(0, 0, (255,0,0))  # same house, for simplicity
codes = [[0]]  # dummy map
h1 = Human(human_id=1, sex="M", x=0, y=0, home=h1_home, codes=codes)
h2 = Human(human_id=2, sex="F", x=1, y=0, home=h2_home, codes=codes)

# give h1 some food
h1.bag = 3
h2.bag = 0

trust = TrustSystem()
print("Before share:", h1.bag, h2.bag, trust.hints)

shared = h1.share_food_with(h2, trust, amount=1, trust_inc=0.01)
print("shared returned?", shared)
print("After share:", h1.bag, h2.bag)
print("TrustSystem state:", trust.hints)