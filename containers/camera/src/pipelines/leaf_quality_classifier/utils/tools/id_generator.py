import string
import random
import datetime


def id_generator(size=8, chars=string.ascii_lowercase + string.digits):
    rng = random.Random(int(str(datetime.datetime.now())[-6:]))
    return ''.join(rng.choice(chars) for _ in range(size))