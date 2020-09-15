from distutils.core import setup


setup(
    name='hopfield-layers',
    author='Hubert Ramsauer, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Lukas Gruber, '
           'Markus Holzleitner, Milena Pavlović, Geir Kjetil Sandve, Victor Greiff, David Kreil, Michael Kopp, '
           'Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter',
    description='Official implementation of continuous Hopfield networks from the Paper '
                'Hopfield Networks is All You Need',
    version='1.0',
    packages=['hopfield_layers'],
    python_requires='>=3.6',
)
