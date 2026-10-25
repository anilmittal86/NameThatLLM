import unittest
import sys

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())
