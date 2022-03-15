import matplotlib.pyplot
import numpy

a = numpy.zeros([3,2])

matplotlib.pyplot.imshow(a, interpolation="nearest")

print(a)