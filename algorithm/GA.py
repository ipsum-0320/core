import sys
sys.dont_write_bytecode = True

from benchmark import ackley_generator, ackley_max, ackley_min

ack = ackley_generator(6)

print(ack([0, 0, 0, 0, 0, 0]))