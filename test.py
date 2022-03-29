import tests
import pkgutil
from os.path import join

def run_all_tests(pkg, prefix=""):
  for test in pkgutil.iter_modules(pkg.__path__):
    importer = test[0]
    test_name = test[1]
    is_pkg = test[2]
    module = importer.find_module(test_name)\
                     .load_module(test_name)
    full_test_name = join(prefix, test_name)
    if is_pkg:
      run_all_tests(module, prefix=full_test_name)
    else:
      print(f"Testing {full_test_name}...")
      module.main(None)

def main():
  run_all_tests(tests)
  print("All tests passed! =)")

if __name__ == '__main__':
  main()
